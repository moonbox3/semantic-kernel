import logging
from contextlib import asynccontextmanager

import uvicorn
from autogen_core import SingleThreadedAgentRuntime
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from samples.demos.process_with_dapr.process.process import get_process
from samples.demos.process_with_dapr.process.steps import CommonEvents, CStepState
from semantic_kernel import Kernel
from semantic_kernel.processes.autogen_runtime.autogen_actor_registration import register_autogen_agents
from semantic_kernel.processes.autogen_runtime.autogen_kernel_process import start as autogen_start
from semantic_kernel.processes.kernel_process.kernel_process_step_state import KernelProcessStepState

logging.basicConfig(level=logging.WARNING)

# Get the kernel and the process (as in your original code)
kernel = Kernel()
process = get_process()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create our SingleThreadedAgentRuntime instead of Dapr
    print("## AutoGen runtime startup ##")
    runtime = SingleThreadedAgentRuntime()

    # Register the agent types with the runtime, injecting the factories you had for steps
    await register_autogen_agents(runtime, kernel, process.factories)

    # Start background message processing
    runtime.start()

    # Expose the runtime so routes can reference it
    app.state.runtime = runtime
    yield

    # On shutdown, stop the runtime
    await runtime.stop_when_idle()
    await runtime.close()
    print("## AutoGen runtime shutdown ##")


# FastAPI app with lifespan
app = FastAPI(title="SKProcess with AutoGen", lifespan=lifespan)


@app.get("/healthz")
async def healthcheck():
    return "Healthy!"


@app.get("/processes/{process_id}")
async def start_process(process_id: str):
    """
    Equivalent to the old /processes/{process_id} Dapr-based route,
    but now uses the SingleThreadedAgentRuntime and AutoGen calls.
    """
    try:
        # Start the process with an initial event
        # (CommonEvents.StartProcess must be a KernelProcessEvent or string)
        context = await autogen_start(
            runtime=app.state.runtime,  # The SingleThreadedAgentRuntime from the lifespan
            process=process,  # Your KernelProcess
            initial_event=CommonEvents.StartProcess,  # The same event used previously
            process_id=process_id,  # The unique process instance ID
        )

        # Once started, retrieve the updated process state
        kernel_process = await context.get_state()

        # Example step usage: find the step named "CStep" and examine its state
        c_step_state: KernelProcessStepState[CStepState] = next(
            (s.state for s in kernel_process.steps if s.state and s.state.name == "CStep"), None
        )

        if c_step_state and c_step_state.state:
            c_step_state_validated = CStepState.model_validate(c_step_state.state)
            print(f"[FINAL STEP STATE]: CStepState current_cycle = {c_step_state_validated.current_cycle}")

        return JSONResponse(content={"processId": process_id}, status_code=200)
    except Exception as ex:
        logging.error(f"Error starting process: {ex}")
        return JSONResponse(content={"error": "Error starting process"}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="error")  # nosec
