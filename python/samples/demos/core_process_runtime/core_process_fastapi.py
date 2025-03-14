# Copyright (c) Microsoft. All rights reserved.

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse

from samples.demos.process_with_dapr.process.process import get_process
from samples.demos.process_with_dapr.process.steps import CommonEvents, CStepState
from semantic_kernel import Kernel
from semantic_kernel.processes.core_runtime.core_kernel_process import start as core_runtime_start
from semantic_kernel.processes.kernel_process.kernel_process_step_state import KernelProcessStepState

logging.basicConfig(level=logging.WARNING)

kernel = Kernel()
process = get_process()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("## Core runtime startup ##")
    yield


app = FastAPI(title="SKProcess with Core Runtime", lifespan=lifespan)


@app.get("/healthz")
async def healthcheck():
    return "Healthy!"


@app.get("/processes/{process_id}")
async def start_process(process_id: str):
    try:
        context = await core_runtime_start(
            process=process,
            initial_event=CommonEvents.StartProcess,
            process_id=process_id,
        )

        kernel_process = await context.get_state()

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
