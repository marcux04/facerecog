@echo off
setlocal

echo === Activando entorno virtual ===
python -m venv venv
call venv\Scripts\activate

echo === Actualizando pip ===
python -m pip install --upgrade pip

echo === Instalando pipwin ===
pip install pipwin

echo === Instalando dlib con pipwin ===
pipwin install dlib

echo === Instalando face_recognition y Pillow ===
pip install face_recognition pillow

echo === Instalación completa ===
pause
