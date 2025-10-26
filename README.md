# 🌿 Consultor RAG de Buenas Prácticas Veterinarias (SENASA)

Este proyecto implementa una aplicación de **Retrieval-Augmented Generation (RAG)** que permite a los usuarios consultar documentos técnicos de Buenas Prácticas Agropecuarias (BPA) de SENASA a través de una interfaz de chat.

El objetivo principal es demostrar la integración de herramientas de PLN y modelos generativos para resolver un problema concreto en el sector agropecuario, sirviendo como una herramienta de consulta rápida y fiable basada en documentación oficial.

Los manuales se obtienen del siguiente [link](https://www.argentina.gob.ar/senasa/publicaciones/manuales-y-guias).

## 🚀 Requisitos y Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```
CHATBOT-MANUALES-V.../
├── chroma_db/                  # 💾 Base de datos vectorial
├── data/                       # 📂 Documentos PDF de origen para RAG
│   └── documents/
├── src/                        # 🧠 Código fuente de la aplicación
│   ├── app.py                  # Aplicación principal Streamlit (Interfaz y RAG en tiempo real)
│   └── data_processor.py       # Script de ingesta: chunking y creación de embeddings
├── venv/                       # 🐍 Entorno virtual de Python
├── .env.example                # Plantilla para variables de entorno
├── .gitignore                  # Reglas para ignorar archivos
├── README.md                   # 📄 Este archivo
└── requirements.txt            # Dependencias del proyecto
```

## 💻 Ejecución Local

Sigue estos pasos para configurar y ejecutar la aplicación en tu máquina local:

### 1\. Requisitos Previos

- Python (versión 3.9+)
- Clave de API de Google Gemini (requerida para el LLM).

### 2\. Configuración del Entorno

1.  **Clonar el repositorio:**

    ```bash
    git clone git@github.com:Isaiasgaray/chatbot-manuales-veterinarios-utn.git
    cd CHATBOT-MANUALES-V...
    ```

2.  **Crear y activar el entorno virtual:**

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar Variables de Entorno:**
    Crea un archivo llamado `.env` guiandose con el `.env.example`

### 3\. Construcción de la Base de Datos Vectorial

Convierte los documentos en vectores que el RAG puede buscar. Solo se necesita ejecutar una vez, o cada vez que se agreguen nuevos documentos.

```bash
python src/data_processor.py
```

Al finalizar, se creará la carpeta `chroma_db/`.

### 4\. Iniciar la Aplicación

Ejecuta la aplicación Streamlit:

```bash
streamlit run src/app.py
```

La aplicación se abrirá automáticamente en tu navegador predeterminado.

---

## 🌐 Acceso a la Versión Desplegada

La aplicación está disponible públicamente en:

- **[Link en Streamlit Cloud](https://chatbot-manuales-veterinarios-utn-ku4gj7bzeed2ohbqbpt3vo.streamlit.app/)**

---

## 🧐 Diseño y Decisiones Técnicas

### Arquitectura RAG

La aplicación sigue el patrón RAG (Retrieval-Augmented Generation) para garantizar que las respuestas del LLM sean precisas y trazables a la documentación fuente.

1.  **Ingesta (`data_processor.py`):**
    - **Loader:** Se usa `PyPDFLoader` para la ingesta de documentos.
    - **Chunking:** Se emplea `RecursiveCharacterTextSplitter` con un `chunk_size` de 1000 y `chunk_overlap` de 200. El solapamiento asegura que el contexto no se pierda en los límites de los fragmentos.
    - **Embeddings:** Se utiliza el modelo **`all-MiniLM-L6-v2`** por su balance entre rendimiento y velocidad.
    - **Vector Store:** Se eligió **ChromaDB** porque es ligero, fácil de configurar (`persist_directory`) y no requiere infraestructura externa, cumpliendo con el espíritu de un MVP.
2.  **Generación (`app.py`):**
    - **LLM:** Se eligió **Gemini-2.5 Flash** por su eficiencia, potencia en razonamiento y su excelente integración a través de `langchain_google_genai`.
    - **Prompting:** El _System Prompt_ define el rol del LLM como un "Consultor de Buenas Prácticas veterinarias de SENASA" y lo **limita estrictamente** a las fuentes proporcionadas, mitigando las alucinaciones.
    - **Interfaz:** **Streamlit** se utiliza para crear la interfaz de chat funcional y minimalista requerida para el MVP.

### Trade-offs y Limitaciones Conocidas

- **Latencia:** La latencia puede variar levemente al invocar a Gemini, especialmente cuando el _retriever_ recupera los 3 fragmentos y el contexto es extenso.
- **Alcance:** El sistema está limitado **solo** a la información contenida en los documentos PDF cargados. Cualquier pregunta fuera de este corpus resultará en una respuesta indicando la falta de información, según las instrucciones del _prompt_.
- **Escalabilidad:** ChromaDB es adecuado para este MVP. Para un escenario empresarial con millones de documentos, se consideraría una base de datos vectorial más escalable como Pinecone o Weaviate.
