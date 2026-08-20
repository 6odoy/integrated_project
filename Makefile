.PHONY: help download-kermany

help:
	@echo "Objetivos disponibles:"
	@echo "  make download-kermany  Descarga y verifica el dataset Chest X-Ray Pneumonia (Kermany)."

download-kermany:
	bash first-installment/scripts/download_kermany.sh
