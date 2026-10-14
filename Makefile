.PHONY: install test api dashboard demo

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest -q

api:
	PYTHONPATH=src uvicorn api:app --reload

dashboard:
	streamlit run app.py

demo:
	python3 src/demo_data.py
