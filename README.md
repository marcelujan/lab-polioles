# Laboratorio de Polioles

Aplicación desarrollada en Streamlit para gestionar muestras y análisis físico-químicos en un entorno tipo Excel.

## 🧪 Funcionalidades

- Carga de muestras con análisis físico-químicos asociados
- Edición de análisis directamente en tabla estilo Excel
- Filtrado y ordenamiento de datos
- Exportación a Excel
- Protección con contraseña
- Configuración externa para modificar la contraseña fácilmente

## 🚀 Cómo usar

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu_usuario/lab-polioles.git
cd lab-polioles
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Crear archivo de configuración

Crear un archivo llamado `config.toml` en la raíz del proyecto con este contenido:

```toml
[auth]
password = "irqplantapiloto"
```

### 4. Ejecutar la app

```bash
streamlit run app.py
```

## 📁 Estructura del proyecto

```
.
├── app.py
├── config.toml
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```
