from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from routers.test.translation import run


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run()
    yield


app = FastAPI(lifespan=lifespan)
