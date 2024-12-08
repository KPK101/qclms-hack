# Models

onnx models for FaceAttribNet - returns attributes for input face
The predictions of this model will be used as source prompts to the LLM assistant.

This will help the LLM understand the user's emotional state  using multimodal information. This creates a more humane experience for the user.

## Future Extensions
- Track history of user state.
- Learn personalized user behavior and response to questions, learn to behave according to user's needs.
- A strategy to do this is to keep a summary doc of a users behavior. To avoid fine-tuning, let the LLM perform RAG on the summary and give most appropriate response.
