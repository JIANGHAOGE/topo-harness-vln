from . import schemas, tools


def register(ctx) -> None:
    registrations = (
        ("cee_navigation_status", schemas.STATUS, tools.navigation_status),
        ("cee_start_agent_task", schemas.START_AGENT_TASK, tools.start_agent_task),
        ("cee_navigation_task_status", schemas.NAVIGATION_TASK_STATUS, tools.navigation_task_status),
        ("cee_cancel_agent_task", schemas.CANCEL_AGENT_TASK, tools.cancel_agent_task),
    )
    for name, schema, handler in registrations:
        ctx.register_tool(name=name, toolset="cee-navigation", schema=schema, handler=handler)
