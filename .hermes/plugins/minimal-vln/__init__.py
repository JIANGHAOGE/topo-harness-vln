from . import schemas, tools


def register(ctx) -> None:
    ctx.register_tool(name="observe", toolset="minimal-vln", schema=schemas.OBSERVE, handler=tools.observe)
    ctx.register_tool(name="step", toolset="minimal-vln", schema=schemas.STEP, handler=tools.step)
