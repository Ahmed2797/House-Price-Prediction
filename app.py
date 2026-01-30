from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from prediction import predict_price


class HouseInput(BaseModel):
	date: str
	bedrooms: int
	bathrooms: float
	sqft_living: int
	sqft_lot: int
	floors: float
	waterfront: int
	view: int
	condition: int
	sqft_above: int
	sqft_basement: int
	yr_built: int
	yr_renovated: int
	city: str
	statezip: str
	country: str
	# optional fields sent by the frontend
	street: Optional[str] = None
	state_code: Optional[int] = None


app = FastAPI(title="House Price Prediction API")

# Allow requests from local frontend/dev servers. Adjust origins for production.
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Serve project files (index.html) under /static
app.mount("/static", StaticFiles(directory=".", html=True), name="static")


@app.get("/health")
def health():
	return {"status": "ok"}


@app.post("/predict")
async def predict(house: HouseInput):
	try:
		# Remove optional fields that weren't in the training set
		payload = house.dict(exclude={'street', 'state_code'})
		result = await run_in_threadpool(predict_price, payload)
		return result
	except Exception as e:
		print(f"Error in predict: {e}")
		import traceback
		traceback.print_exc()
		raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



@app.get("/")
def root():
	return RedirectResponse(url="/static/index.html")


@app.get("/favicon.ico")
def favicon():
	return Response(status_code=204)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("app:app", host="0.0.0.0", port=8000)


## uvicorn app:app --reload

{
  "date": "2014-10-13",
  "bedrooms": 4,
  "bathrooms": 2.5,
  "sqft_living": 2770,
  "sqft_lot": 5650,
  "floors": 2,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "sqft_above": 2770,
  "sqft_basement": 0,
  "yr_built": 1989,
  "yr_renovated": 0,
  "street": "123 Main St",
  "city": "Seattle",
  "statezip": "WA 98103",
  "country": "USA"
}
