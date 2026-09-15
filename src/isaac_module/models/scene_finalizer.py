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
        """
        SimManager.get().materialise_components(
            materialise_workcell(generic_dependencies(dependencies), logger=self.logger)
        )
        SimManager.get().finalize_scene()
