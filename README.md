NyaySathi — Legal Aid Chatbot
A Human-Friendly Guide
NyaySathi (न्याय साथी) is a free, open-source chatbot designed to be a supportive companion for individuals seeking general legal information in India. It was built using a combination of traditional information retrieval methods and modern AI. The bot provides helpful, informative answers to a wide range of common legal queries, but it is crucial to remember that its responses are for informational purposes only and do not constitute legal advice.

This chatbot was created with the core principle of accessibility. It does not require a compulsory API key to function. Its primary knowledge-based retrieval engine is a robust TF-IDF (Term Frequency-Inverse Document Frequency) model built using the popular scikit-learn library. This model has been trained on a comprehensive knowledge base of over 100 legal queries, allowing it to provide relevant and accurate answers without external services. For those who want to enhance the clarity and readability of the bot's responses, it offers an optional integration with OpenAI's large language models, like GPT-4o-mini, which can be enabled by providing your own API key. This hybrid approach ensures the bot is both self-sufficient and capable of delivering polished, high-quality responses when needed.

Key Features
Offline Knowledge Base: The chatbot's core functionality relies on a pre-trained TF-IDF model, making it highly efficient and independent of external API calls. This allows it to answer more than 100 legal queries.

Extensible Knowledge: The bot's knowledge can be easily expanded by uploading a simple CSV file with new legal questions and answers, allowing it to adapt and grow over time.

Optional AI Polish: While it works perfectly fine on its own, you have the choice to connect an OpenAI API key to use a powerful LLM to rewrite and clarify the bot's answers, making them more human-like and easier to understand.

Safety First: The chatbot is designed with a strong safety layer that detects and refuses to assist with any queries related to illegal activities, violence, or harmful actions.

User-Friendly Interface: Built with the Streamlit framework, the bot's interface is clean, modern, and easy to navigate. It allows for quick queries, includes a chat history, and provides the option to view the sources of its answers.

NyaySathi represents a blend of classic data science and modern AI, providing a reliable and responsible resource for general legal information without the need for constant internet access or paid services. It's a true testament to the power of open-source development in providing access to justice-related information for everyone
