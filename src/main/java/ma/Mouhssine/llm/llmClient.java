package ma.Mouhssine.llm;


import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;

import jakarta.enterprise.context.Dependent;


@Dependent
public class llmClient {
    // Interface de l’assistant : une méthode pour dialoguer avec le LLM
    interface Assistant {
        String chat(String userMessage);
    }

    public static void main(String[] args) {

        // 1. Récupération de la clé API Gemini depuis les variables d’environnement
        String llmKey = System.getenv("GEMINI_API_KEY");
        if (llmKey == null) {
            System.err.println("GEMINIKEY non définie !");
            return;
        }

        // 2. Création du modèle LLM (Gemini)
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.5-flash") // modèle rapide et récent
                .temperature(0.3)              // température faible = réponses plus précises
                .maxOutputTokens(512)          // limite du nombre de tokens générés
                .build();

    }
}