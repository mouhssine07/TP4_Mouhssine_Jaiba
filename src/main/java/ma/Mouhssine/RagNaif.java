package ma.Mouhssine;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class RagNaif {


    public static void main(String[] args) {

        try {
            Path path = Paths.get("src/main/resources/ml.pdf");
            System.out.println("Chargement du fichier : " + path);

            ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();

            Document document = FileSystemDocumentLoader.loadDocument(path, parser);
            System.out.println("Document chargé avec succès.");

            DocumentSplitter splitter = DocumentSplitters.recursive(500, 50); // 500 tokens par segment, 50 de chevauchement
            List<TextSegment> segments = splitter.split(document);
            System.out.println("Nombre de segments créés : " + segments.size());

            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
            List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

            System.out.println("Embeddings générés : " + embeddings.size());


            EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
            embeddingStore.addAll(embeddings, segments);

            System.out.println("💾 Embeddings enregistrés dans le store en mémoire ✅");

        }catch (Exception e) {
            e.printStackTrace();
        }

    }

}
