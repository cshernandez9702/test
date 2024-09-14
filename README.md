# Construcción de un Chatbot con Generación Aumentada por Recuperación (RAG) usando Llama 3 para Consultas Locales en PDFs

## Introducción

En la actualidad, las organizaciones necesitan extraer información rápidamente de grandes volúmenes de documentos, incluyendo PDFs confidenciales. Los métodos tradicionales pueden ser ineficientes y propensos a errores, como las alucinaciones en modelos de lenguaje. Para mejorar la productividad y reducir estas alucinaciones, se desarrolla un chatbot que utiliza la **Generación Aumentada por Recuperación (RAG)** y **Llama 3** para permitir consultas eficientes y seguras de documentos PDF locales.

<p align="center">
  <img src="https://github.com/cshernandez9702/mmia_usfq_nlp/blob/main/1.png" alt="Diagrama de Flujo del Proceso RAG" width="400">
</p>


### Generación Aumentada por Recuperación (RAG)

**RAG** es un método que permite recuperar documentos relevantes de una base de conocimiento y los utilizarlos como contexto para generar respuestas a las consultas de los usuarios.

<figure style="text-align: center;">
  <img src="https://github.com/cshernandez9702/mmia_usfq_nlp/blob/main/2.png" alt="Diagrama de Flujo del Proceso RAG" width="1200">
  <figcaption>
    <em>Diagrama de Flujo del Proceso RAG. Fuente: <a href="https://blog.langchain.dev/semi-structured-multi-modal-rag/">Enlace a la fuente</a></em>
  </figcaption>
</figure>


### RAG y Llama 3

Al integrar RAG con Llama 3, mejoramos la capacidad del modelo para generar respuestas que son tanto contextualmente relevantes como informadas por el contenido específico de los PDFs locales. Esta combinación permite construir un chatbot capaz de manejar consultas complejas sobre documentos confidenciales sin comprometer la seguridad de los datos.

## Objetivos

- **Desarrollar un chatbot** que pueda consultar documentos PDF locales utilizando RAG y Llama 3.
- **Garantizar la confidencialidad de los datos** procesando los documentos localmente sin enviar datos a servidores externos.
- **Proporcionar respuestas precisas y contextualmente relevantes** a las consultas de los usuarios.


- **Modelos**:
  - Llama 3 (8B parámetros)
  - Sentence Transformers para embeddings
- **Datos**:
  - Documentos PDF locales


### Visión General

Componentes:

1. **Carga y Configuración del Modelo**: Carga del modelo Llama 3 y su tokenizador.
2. **Carga y Procesamiento de PDFs**: Lectura de PDFs y extracción de contenido textual.
3. **Generación de Embeddings y Configuración del Vector Store**: Conversión de texto en embeddings y almacenamiento usando Chroma.
4. **Configuración de RAG**: Creación de un RAG CHAIN por recuperación.


### Carga y Configuración del Modelo

Comenzamos cargando el modelo Llama 3 y su tokenizador. El modelo se configura para ejecutarse eficientemente en el hardware disponible, utilizando aceleración GPU si es posible.

```python
def load_model_and_tokenizer(model_path):
    start_time = time()
    config = transformers.AutoConfig.from_pretrained(model_path,
                                                     trust_remote_code=True,
                                                     max_new_tokens=2048)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                              trust_remote_code=True,
                                                              config=config,
                                                              #quantization_config=quant_config,
                                                              device_map='auto' )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model and tokenizer loaded in {round(time() - start_time, 3)} sec.")
    return model, tokenizer
 ```



- **Cuantización**: Aunque consideramos la cuantización de 8 bits para reducir el uso de memoria, optamos por cargar el modelo completo ya que disponíamos de recursos suficientes.
- **Mapeo de Dispositivos**: El modelo se asigna automáticamente a los dispositivos disponibles (CPU o GPU).

### Carga y Procesamiento de PDFs usando la librería `unstructured`

Para procesar los documentos PDF, utilizamos la librería `unstructured`, que permite extraer el texto de los PDFs sin incluir imágenes. Este paso es necesario para crear los embeddings.


```python
# Cargar el archivo PDF usando unstructured (sin imágenes)
def load_pdf_with_unstructured(pdf_path):
    raw_pdf_elements = partition_pdf(
        filename=pdf_path,                  # Ruta al archivo PDF
        strategy="hi_res",                  # Estrategia de alta resolución
        extract_images_in_pdf=False,        # No extraer imágenes dentro del PDF
        extract_image_block_types=["Table"] # Extraer solo las tablas
    )

    # Inicializar listas para cada tipo de contenido
    Header, Footer, Title, NarrativeText, Text, ListItem, Tables = [], [], [], [], [], [], []

    # Clasificar los elementos del PDF
    for element in raw_pdf_elements:
        element_type = str(type(element))
        if "Header" in element_type:
            Header.append(str(element))
        elif "Footer" in element_type:
            Footer.append(str(element))
        elif "Title" in element_type:
            Title.append(str(element))
        elif "NarrativeText" in element_type:
            NarrativeText.append(str(element))
        elif "Text" in element_type:
            Text.append(str(element))
        elif "ListItem" in element_type:
            ListItem.append(str(element))
        elif "Table" in element_type:
            Tables.append(str(element))  # Almacenar las tablas como texto

    # Combinar los elementos que nos interesan para embeddings
    combined_text = "\n".join(NarrativeText + Text + ListItem)
    combined_tables = "\n".join(Tables)  # Combinar las tablas como texto adicional
    return combined_text, combined_tables
 ```



- **Extracción de Texto**: texto narrativo, texto regular, elementos de lista y tablas.
- **Exclusión de Imágenes**: Se excluyen las imágenes debido a que era necesario un modelo adicional para poder procesarlas. 
- **Procesamiento**: Clasificamos y combinamos los diferentes tipos de contenido para su posterior procesamiento.

### Generación de Embeddings y Configuración del Vector Store

Generamos embeddings a partir del texto extraído utilizando un modelo pre-entrenado de Sentence Transformer y los almacenamos usando ChromaDB.



```python
def setup_vectorstore_unstructured(doc_text, table_text, model_name="sentence-transformers/all-mpnet-base-v2"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

    # Usar RecursiveCharacterTextSplitter para dividir el texto y las tablas
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks_text = splitter.split_text(doc_text)
    chunks_tables = splitter.split_text(table_text)

    # Crear los documentos en el formato esperado por Chroma
    documents = [Document(page_content=chunk, metadata={"source": "unstructured_pdf_text"}) for chunk in chunks_text]
    table_documents = [Document(page_content=chunk, metadata={"source": "unstructured_pdf_table"}) for chunk in chunks_tables]

    # Combinar los documentos de texto y tablas
    all_documents = documents + table_documents

    # Crear el vector store a partir de los documentos
    return Chroma.from_documents(all_documents, embeddings, persist_directory="./NLP_FINAL/chroma_db_unstructured")
 ```

- **División de Texto**: El texto se divide en fragmentos manejables para la generación de embeddings.
- **Embeddings**: Los embeddings  capturan el significado semántico.
- **Vector Store**: Chroma almacena los embeddings y facilita la búsqueda de similitud durante la recuperación.

<figure style="text-align: center ;">
  <img src="https://github.com/cshernandez9702/mmia_usfq_nlp/blob/main/3.png" alt="" width="400">
  <figcaption>
    <em>Diagrama de Flujo del Proceso RAG. Fuente: <a href="https://tech-depth-and-breadth.medium.com/my-notes-from-deeplearning-ais-course-on-advanced-retrieval-for-ai-with-chroma-2dbe24cc1c91">Enlace a la fuente</a></em>
  </figcaption>
</figure>



### Chatbot

Implementamos una clase de chatbot que mantiene un historial de conversación y maneja las interacciones con el usuario.

```python
class RAGChatbot:
    def __init__(self, qa_chain, max_context_length=4096):
        self.qa_chain = qa_chain
        self.history = ""  # Historial de preguntas y respuestas
        self.max_context_length = max_context_length  # Controlar el tamaño del contexto

    def add_to_history(self, question, answer):
        """
        Añadir la pregunta y la respuesta al historial.
        """
        self.history += f"Question: {question}\nAnswer: {answer}\n\n"

        # Controlar la longitud del historial para que no sobrepase el máximo del modelo
        if len(self.history) > self.max_context_length:
            # Recortar el historial si es muy largo
            self.history = self.history[-self.max_context_length:]

    def get_chatbot_response(self, new_question):
        """
        Genera una respuesta a una nueva pregunta utilizando el historial como contexto.
        """
        # Concatenar el historial con la nueva pregunta
        input_with_history = f"{self.history}\nQuestion: {new_question}"

        # Obtener la respuesta del sistema RAG
        start_time = time()
        raw_response = self.qa_chain.run(input_with_history)
        elapsed_time = round(time() - start_time, 3)

        # Filtrar la respuesta usando el delimitador 'Answer:'
        final_response = filter_delimited_response(raw_response, new_question, delimiter="Answer:")

        # Añadir la nueva pregunta y respuesta al historial
        self.add_to_history(new_question, final_response)

        # Mostrar la respuesta
        print(f"{final_response}\n\nTime taken: {elapsed_time} sec.\n")
        return final_response

    def chat(self):
        """
        Inicia el modo interactivo del chatbot, solicitando preguntas del usuario.
        """
        print("Bienvenido al chatbot. Escribe 'salir' para terminar la conversación.")

        while True:
            # Solicitar input del usuario
            user_question = input("Tú: ")

            # Verificar si el usuario desea terminar la conversación
            if user_question.lower() in ['salir', 'exit']:
                print("Chat terminado.")
                break

            # Generar y mostrar la respuesta del chatbot
            self.get_chatbot_response(user_question)

# Crear una instancia del chatbot
chatbot = RAGChatbot(qa_chain=rag_chain)

# Iniciar la interacción con el chatbot
chatbot.chat()

 ```

- **Historial**: Mantiene un registro de interacciones previas para proporcionar contexto.
- **Bucle Interactivo**: Solicita continuamente entradas del usuario y genera respuestas.

- **Tokens Máximos**: Limitado a 4096 tokens de conexto, establecido para asegurar que las respuestas sean concisas y dentro de las limitaciones del modelo.
- **Tamaño y Solapamiento de Fragmentos**: Chunks de 1000 Tokens con solapamientos de 100 Tokens.

## Resultados

El chatbot responde exitosamente a consultas recuperando información relevante de los PDFs locales y generando respuestas coherentes.

### Ejemplo de Interacción

**Usuario**: "¿Qué significa el término "Cámara de Compensación Especializada (CCE)" segun el manual operativo?"

**Chatbot**: "In this context, the "Cámara de Compensación Especializada (CCE)" refers to a specialized compensation chamber within the Banco Central Europeo (BCE) that is responsible for processing and settling transactions between the entities participating in the Red Coonecta network. The CCE is responsible for ensuring that the transactions are properly compensated and settled, and that the entities are accurately credited or debited for their transactions.assistant"

**Tiempo Tomado**: 11.985 seg

### Ejemplo de Interacción

**Usuario**: "¿Qué se menciona acerca de las modalidades de integración en la red de cajeros automáticos?"

**Chatbot**: "According to the provided context, the entity participating in the Red de Cajeros Automáticos (RCA) network has two modalities of integration, which are selected by each entity in accordance with its technical, technological, and operational capacity. The two modalities are:

1. Procesamiento en línea: This modality involves the processing of transactions in real-time, allowing for immediate settlement and clearing of transactions.
2. Conciliación, compensación y liquidación: This modality involves the reconciliation, compensation, and settlement of transactions, which is done through the Cámara de Compensación
"

**Respuestas correcta: ** Debia contener definiciones de Frontend, Backend y Switch Externo.

**Tiempo Tomado**: 12.55 seg


- **Tiempo de Respuesta**: El tiempo promedio de respuesta es de 7 a 15 segundos. 
- **Precisión**: Las respuestas son contextualmente relevantes y precisas basadas en los PDFs proporcionados.


## Métricas

### Faithfulness
Faithfulness evalúa si la respuesta generada por el modelo es fiel a la información contenida en los documentos de contexto recuperados. En otras palabras, verifica si la respuesta generada no introduce información incorrecta o inventada que no está presente en el contexto proporcionado.

### Answer Correctness
Answer Correctness mide qué tan correcta es la respuesta generada en comparación con la respuesta esperada o ground_truth. Evalúa si la respuesta generada es precisa y directamente responde a la pregunta.

| index | faithfulness | answer_correctness | Pregunta                                                                                          |
|-------|--------------|--------------------|--------------------------------------------------------------------------------------------------|
| 0     | 0.714286     | 0.762032           | ¿Cuál es la visión de Corebi en cuanto a sus servicios de Data & Analytics?                       |
| 1     | 0.750000     | 0.551758           | ¿Qué es compensación según el manual operativo?                                                   |
| 2     | 0.000000     | 0.709703           | ¿Qué entidad financiera ejecuta los procesos de liquidación en la Red Coonecta?                   |
| 3     | 1.000000     | 0.220632           | ¿Qué significa la sigla RTC en el contexto de la Red Coonecta?                                    |
| 4     | 0.666667     | 0.958903           | ¿Qué es un cajero automático?                                                                     |
| 5     | 0.833333     | 0.677557           | ¿Qué es un tarjetahabiente?                                                                       |
| 6     | 0.846154     | 0.853127           | ¿Cuál es la función del switch central RTC en la red de cajeros automáticos?                      |
| 7     | 0.800000     | 0.213479           | ¿Qué responsabilidades tienen las entidades participantes en el proceso de conciliación?          |
| 8     | 0.400000     | 0.674976           | ¿Cuál es el rol del Banco Central del Ecuador en el proceso de liquidación de la Red de Cajeros Automáticos? |


## Conclusión

se desempeña correctamente al consultar PDFs locales, superando en general el 70% en 'faithfulness' y 'answer correctness'. El sistema garantiza la confidencialidad de los datos al procesar los documentos localmente y proporciona respuestas precisas y contextualmente relevantes. Sin embargo, es necesario mejorar en el procesamiento de texto, específicamente en la división de los documentos en 'chunks' y en la tokenización, para optimizar aún más la calidad y coherencia de las respuestas.


## Referencias

- [Documentación de Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Documentación de LangChain](https://langchain.readthedocs.io/en/latest/)
- [Documentación de Chroma](https://www.trychroma.com/)
- [Librería unstructured](https://github.com/Unstructured-IO/unstructured)
- [Librería Ragas](https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html)
- [Librería Ragas](https://docs.ragas.io/en/latest/concepts/metrics/answer_correctness.html)
- [Librería unstructured](https://github.com/Unstructured-IO/unstructured)
- [Multimodal RAG](https://github.com/sunnysavita10/Generative-AI-Indepth-Basic-to-Advance/blob/main/MultiModal%20RAG/Extract_Image%2CTable%2CText_from_Document_MultiModal_Summrizer_RAG_App.ipynb)
- [LLama 3, LangChain](https://www.kaggle.com/code/gpreda/rag-using-llama3-langchain-and-chromadb)
  
