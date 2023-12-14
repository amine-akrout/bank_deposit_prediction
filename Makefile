train-model:
	@echo "Starting Prefect server..."
	prefect server start &
	@echo "Server started. Running model.py..."
	sleep 5
	python model.py
	@echo "model.py script run successfully."