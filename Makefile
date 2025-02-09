train-model:
	@echo "Starting Prefect server..."
	prefect server start &
	@echo "Server started. Running model.py..."
	sleep 5
	python model.py
	@echo "model.py script run successfully."

minikube-start:
	@echo "Starting minikube..."
	minikube start --driver=docker --memory=8192 --cpus=4

minikube-ingress:
	minikube addons enable ingress
	minikube addons enable ingress-dns

minikube-dns:
	Add-DnsClientNrptRule -Namespace ".test" -NameServers "$(minikube ip)"

mininkube-local-registry:
  	minikube docker-env | Invoke-Expression
# eval $(minikube docker-env)
	@echo "Building api image..."
	docker build -t my-api-image .
	@echo "Building airflow image..."
	docker build -t my-airflow-image ./airflow

minikube-deploy:
	kubectl apply -f k8s/

minikube-dashboard:
	minikube dashboard

minikube-stop:
	@echo "Stopping minikube..."
	minikube stop
	minikube delete
