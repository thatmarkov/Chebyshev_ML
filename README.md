# Chebyshev\_ML

**Русский (Russian)** | [English](#english)

## Русский
Обучение нейронной сети для регрессии полиномов Чебышёва для измерения и компенсации фазовых искажений по картинам дифракции на случайных фазовых масках, выведенных на ПВМС. Содержит файлы для генерации синтетического датасета, а также файлы для обработки экспериментальных данных дял верификации метода.

Разработано в рамках научной работы для улучшения качества восстановления изображений с фазовых голограмм.

- `dataSetCreationScenario.ipynb поэтапно генерирует набор пучков с аберрациями и рассчитывает набор трехканальных изображений (1 канал - 1 картина дифракции на фазовой маске).`

- `Training_v_1_5_1.ipynb осуществляет обучение свёрточной нейронной сети на основе архитектуры ResNet18.`

Проект находится в работе, в процессе - валидация метода на экспериментальном стенде.

## English
Training a neural network for regression of Chebyshev polynomials to measure and compensate phase distortions based on an analysis of diffraction patterns produced by random phase masks displayed on a spatial light modulator (SLM). The repository contains scripts for generating a synthetic dataset, as well as utilities for processing experimental data to verify the method.

This work is part of a research project aimed at improving the quality of image reconstruction from phase holograms.

- `dataSetCreationScenario.ipynb` – step-by-step generation of a set of aberrated beams and calculation of a three-channel image dataset (each channel corresponds to a diffraction pattern on a phase mask).
- `Training_v_1_5_1.ipynb` – training a convolutional neural network based on the ResNet18 architecture.

The project is currently under active development; experimental validation on an optical stand is in progress.
