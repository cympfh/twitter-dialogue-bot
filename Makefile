## training mode (newly)
train:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python bot.py train_autoencoder --name $(shell date "+%Y-%m-%d-%s")

## training mode (resume from a snapshot)
train-from:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python bot.py train --name $(shell date "+%Y-%m-%d-%s") --resume $(shell ls -1 snapshots/*.h5|peco)

lint:
	mypy --ignore-missing-imports .

.DEFAULT_GOAL := help

## shows this
help:
	@grep -A1 '^## ' ${MAKEFILE_LIST} | grep -v '^--' |\
		sed 's/^## *//g; s/:$$//g' |\
		awk 'NR % 2 == 1 { PREV=$$0 } NR % 2 == 0 { printf "\033[32m%-18s\033[0m %s\n", $$0, PREV }'
