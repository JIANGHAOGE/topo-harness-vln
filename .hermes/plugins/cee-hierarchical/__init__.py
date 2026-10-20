from . import image_context, request_normalization, schemas, tools


def register(ctx) -> None:
    ctx.register_middleware("llm_request", image_context.prune_historical_navigation_images)
    ctx.register_middleware("tool_request", request_normalization.normalize_navigation_request)
    registrations = (
        ("begin_navigation_task", schemas.BEGIN_NAVIGATION_TASK, tools.begin_navigation),
        ("recall_navigation_memory", schemas.RECALL_NAVIGATION_MEMORY, tools.recall_navigation_memory),
        ("advance_navigation", schemas.ADVANCE_NAVIGATION, tools.advance_navigation),
        ("finish_navigation_task", schemas.FINISH_TASK, tools.finish_task),
    )
    for name, schema, handler in registrations:
        ctx.register_tool(name=name, toolset="cee-hierarchical", schema=schema, handler=handler)
