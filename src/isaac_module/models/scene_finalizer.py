"""viam:isaac-sim-devin:scene-finalizer - signals completed scene population.

Configure one per machine whose world sets ``wait_for_finalizer``, with
``depends_on`` naming the world component and every arm, gripper, camera and
base in the scene. viam-server then builds this component last, and its
construction tells the sim the scene is fully populated and it is safe to
start stepping. Until POST_FINALIZER_WARMUP_STEPS steps complete after that,
operational calls on other components answer UNAVAILABLE with "Isaac Sim is
initializing; retry shortly". A world that does not set ``wait_for_finalizer``
never watches for a finalizer, so configuring one has no effect there.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from typing import ClassVar

from typing_extensions import Self
from viam.components.generic import Generic
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..frame_system import frame_system_colliders
from ..sim_manager import SimManager
from ..workcell_client import generic_dependencies, materialise_workcell


class IsaacSceneFinalizer(Generic, EasyResource):  # type: ignore[misc]  # SDK: API is Final on the component, redeclared by EasyResource
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "scene-finalizer")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        finalizer = cls(config.name)
        finalizer.reconfigure(config, dependencies)
        return finalizer

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        depends_on = list(config.depends_on)
        if not depends_on:
            raise ValueError(
                "scene-finalizer needs depends_on naming the world component and every "
                "arm, gripper, camera and base, or it may be built before the scene is "
                "populated"
            )
        return depends_on, []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        """Materialises every ``viam:workcell-components`` dependency's
        scenery before signalling the scene complete.

        Every ``rdk:component:generic`` dependency is probed
        (``workcell_client.generic_dependencies`` and
        ``materialise_workcell``'s own ``get_schema`` check), so a
        dependency that is not workcell scenery, this cell's own world
        included, is skipped with no config of its own required to tell it
        apart. A cell with no workcell components in its dependencies
        materialises nothing and behaves exactly as before this capability
        existed.

        Runs on a background thread and returns at once. Gathering a full
        workcell is dozens of DoCommand round-trips to sibling components, and
        doing that inline blocks the module's event loop for the whole time, so
        the module answers no RPC at all while it runs. viam-server allows five
        seconds for a module to answer ValidateConfig, times out, marks this
        resource failed and rebuilds it, which starts the gathering over: a
        loop that never converges. Ordering is not what the inline call was
        protecting. The world blocks on ``SimManager``'s own scene-finalized
        event, set by ``finalize_scene`` below, whenever that lands.
        """
        resources = generic_dependencies(dependencies)
        # The module's own loop owns every dependency client's channel, so the
        # gather has to be scheduled onto it rather than onto a private one.
        # reconfigure runs inside the module's loop thread, so this is it.
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None  # mock and tests: plain fakes with no channel behind them

        def materialise() -> None:
            try:
                SimManager.get().materialise_frame_system(
                    frame_system_colliders(logger=self.logger, loop=loop)
                )
                SimManager.get().materialise_components(
                    materialise_workcell(resources, logger=self.logger, loop=loop)
                )
            # a component that stops answering must not strand the world, which
            # waits on finalize_scene forever and would otherwise never boot
            except Exception:
                self.logger.exception(
                    "materialising workcell scenery failed; finalizing the scene anyway, "
                    "so the world boots with whatever scenery did land"
                )
            finally:
                SimManager.get().finalize_scene()

        self._materialise_thread = threading.Thread(
            target=materialise, name="scene-finalizer", daemon=True
        )
        self._materialise_thread.start()

    def wait_until_materialised(self, timeout_s: float | None = None) -> None:
        """Test-only join on the background materialise started by
        ``reconfigure``. Production never waits on it: the world already
        blocks on the scene-finalized event that materialise sets."""
        thread = getattr(self, "_materialise_thread", None)
        if thread is not None:
            thread.join(timeout_s)
