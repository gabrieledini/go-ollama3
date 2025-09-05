package main

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/ledongthuc/pdf"
)

// Strutture dati
type Document struct {
	ID      string    `json:"id"`
	Content string    `json:"content"`
	Page    int       `json:"page"`
	Vector  []float64 `json:"vector"`
}

type VectorStore struct {
	Documents []Document `json:"documents"`
	ModelName string     `json:"model_name"`
}

type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type OllamaRequestAdvanced struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	Temperature float64 `json:"temperature,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	NumCtx      int     `json:"num_ctx,omitempty"`
}

type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

type EmbeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type EmbeddingResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

type RAGChatbot struct {
	vectorStore   *VectorStore
	ollamaBaseURL string
	embedModel    string // Modello leggero per embedding
	chatModel     string // Modello più ricco per rispondere
	dbPath        string
}

// Inizializza il chatbot con modelli ottimizzati
func NewRAGChatbot() *RAGChatbot {
	return &RAGChatbot{
		vectorStore:   &VectorStore{Documents: []Document{}},
		ollamaBaseURL: "http://localhost:11434",
		// Modello leggero per embedding - ottimo per documenti tecnici italiani
		embedModel: "nomic-embed-text", // Alternativa: "mxbai-embed-large"
		// Modello più ricco per generazione risposte
		chatModel: "gemma3n:e2b", // Alternativa: "llama3.2:1b" "phi3.5:latest" "qwen2.5:3b" "gemma2:2b" "gemma3n:e2b" "gemma3:1b"
		dbPath:    "vectorstore.json",
	}
}

// Configura i modelli
func (r *RAGChatbot) SetModels(embedModel, chatModel string) {
	r.embedModel = embedModel
	r.chatModel = chatModel
	fmt.Printf("🔧 Modelli configurati:\n")
	fmt.Printf("   📝 Embedding: %s\n", embedModel)
	fmt.Printf("   💬 Chat: %s\n", chatModel)
}

// Estrae testo dal PDF con miglioramenti per manuali tecnici
func (r *RAGChatbot) ExtractTextFromPDF(filename string) ([]string, error) {
	file, reader, err := pdf.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("errore apertura PDF: %v", err)
	}
	defer file.Close()

	var pages []string
	totalPages := reader.NumPage()

	fmt.Printf("📄 Elaborazione PDF: %d pagine trovate\n", totalPages)

	for pageNum := 1; pageNum <= totalPages; pageNum++ {
		page := reader.Page(pageNum)
		if page.V.IsNull() {
			continue
		}

		content, err := page.GetPlainText(nil)
		if err != nil {
			log.Printf("Errore estrazione pagina %d: %v", pageNum, err)
			continue
		}

		// Pulisci il testo preservando la struttura tecnica
		cleanContent := r.cleanTechnicalText(content)
		if len(cleanContent) > 30 { // Soglia più bassa per contenuti tecnici
			pages = append(pages, cleanContent)
		}
	}

	return pages, nil
}

// Pulizia ottimizzata per testi tecnici
func (r *RAGChatbot) cleanTechnicalText(text string) string {
	// Preserva numerazioni, codici e riferimenti tecnici
	text = strings.TrimSpace(text)

	// Normalizza spazi multipli ma preserva struttura
	reg := regexp.MustCompile(`\s{3,}`)
	text = reg.ReplaceAllString(text, " ")

	// Preserva caratteri tecnici importanti
	reg = regexp.MustCompile(`[^\p{L}\p{N}\p{P}\p{Z}\-_./\\()[\]{}]+`)
	text = reg.ReplaceAllString(text, " ")

	// Rimuovi spazi eccessivi ma preserva struttura di paragrafi
	reg = regexp.MustCompile(`\n\s*\n\s*\n`)
	text = reg.ReplaceAllString(text, "\n\n")

	return strings.TrimSpace(text)
}

// Suddivisione intelligente per documenti tecnici
func (r *RAGChatbot) ChunkTechnicalText(text string, chunkSize int, overlap int) []string {
	// Prima dividi per paragrafi naturali
	paragraphs := strings.Split(text, "\n\n")

	var chunks []string
	var currentChunk strings.Builder
	wordCount := 0

	for _, paragraph := range paragraphs {
		paragraphWords := strings.Fields(paragraph)

		// Se il paragrafo da solo supera la dimensione chunk
		if len(paragraphWords) > chunkSize {
			// Salva chunk corrente se non vuoto
			if wordCount > 0 {
				chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
				currentChunk.Reset()
				wordCount = 0
			}

			// Dividi il paragrafo lungo
			subChunks := r.splitLongParagraph(paragraph, chunkSize, overlap)
			chunks = append(chunks, subChunks...)
			continue
		}

		// Se aggiungere questo paragrafo supererebbe la dimensione
		if wordCount+len(paragraphWords) > chunkSize && wordCount > 0 {
			chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
			currentChunk.Reset()
			wordCount = 0
		}

		if currentChunk.Len() > 0 {
			currentChunk.WriteString("\n\n")
		}
		currentChunk.WriteString(paragraph)
		wordCount += len(paragraphWords)
	}

	// Aggiungi ultimo chunk se non vuoto
	if wordCount > 0 {
		chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
	}

	return chunks
}

// Suddivide paragrafi lunghi preservando il senso
func (r *RAGChatbot) splitLongParagraph(text string, chunkSize int, overlap int) []string {
	words := strings.Fields(text)
	var chunks []string

	for i := 0; i < len(words); i += chunkSize - overlap {
		end := i + chunkSize
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[i:end], " ")
		chunks = append(chunks, chunk)

		if end == len(words) {
			break
		}
	}

	return chunks
}

// Genera embedding tramite Ollama
func (r *RAGChatbot) GetEmbedding(text string) ([]float64, error) {
	reqBody := EmbeddingRequest{
		Model: r.embedModel,
		Input: text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(r.ollamaBaseURL+"/api/embed", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("errore chiamata Ollama embed: %v", err)
	}
	defer resp.Body.Close()

	var embedResp EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, err
	}

	if len(embedResp.Embeddings) == 0 {
		return nil, fmt.Errorf("nessun embedding ricevuto")
	}

	return embedResp.Embeddings[0], nil
}

// Calcola similarità coseno
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (normA * normB)
}

// Elabora PDF ottimizzato per manuali tecnici
func (r *RAGChatbot) ProcessPDF(filename string) error {
	fmt.Println("📄 Estrazione testo dal PDF...")
	pages, err := r.ExtractTextFromPDF(filename)
	if err != nil {
		return err
	}

	fmt.Printf("✅ Estratte %d pagine\n", len(pages))

	r.vectorStore.Documents = []Document{}

	fmt.Println("🔤 Creazione chunks intelligenti e embedding...")
	totalChunks := 0

	for pageNum, pageText := range pages {
		// Chunking intelligente per testi tecnici
		chunks := r.ChunkTechnicalText(pageText, 400, 75) // Chunks più grandi per contenuto tecnico

		for chunkIdx, chunk := range chunks {
			if len(strings.TrimSpace(chunk)) < 30 {
				continue
			}

			// Genera ID unico
			hasher := md5.New()
			hasher.Write([]byte(chunk))
			docID := fmt.Sprintf("page_%d_chunk_%d_%x", pageNum+1, chunkIdx, hasher.Sum(nil)[:4])

			fmt.Printf("🔄 Processando chunk %d (pagina %d)\r", totalChunks+1, pageNum+1)

			// Genera embedding con modello leggero
			vector, err := r.GetEmbedding(chunk)
			if err != nil {
				log.Printf("Errore embedding per chunk %s: %v", docID, err)
				continue
			}

			doc := Document{
				ID:      docID,
				Content: chunk,
				Page:    pageNum + 1,
				Vector:  vector,
			}

			r.vectorStore.Documents = append(r.vectorStore.Documents, doc)
			totalChunks++

			// Pausa ridotta per modello embedding leggero
			time.Sleep(50 * time.Millisecond)
		}
	}

	r.vectorStore.ModelName = r.embedModel
	fmt.Printf("\n✅ Creati %d chunks con embedding\n", totalChunks)

	return r.SaveVectorStore()
}

// Elabora file TXT ottimizzato
func (r *RAGChatbot) ProcessTXT(filename string) error {
	fmt.Println("📝 Lettura file TXT...")

	content, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("errore lettura file TXT: %v", err)
	}

	text := string(content)
	cleanedText := r.cleanTechnicalText(text)

	if len(strings.TrimSpace(cleanedText)) < 50 {
		return fmt.Errorf("file TXT troppo corto o vuoto")
	}

	fmt.Printf("✅ File letto: %d caratteri\n", len(cleanedText))

	r.vectorStore.Documents = []Document{}
	fmt.Println("🔤 Creazione chunks intelligenti e embedding...")

	// Chunking intelligente per testi tecnici
	chunks := r.ChunkTechnicalText(cleanedText, 400, 75)
	fmt.Printf("📊 Creati %d chunks\n", len(chunks))

	for chunkIdx, chunk := range chunks {
		if len(strings.TrimSpace(chunk)) < 30 {
			continue
		}

		hasher := md5.New()
		hasher.Write([]byte(chunk))
		docID := fmt.Sprintf("txt_chunk_%d_%x", chunkIdx, hasher.Sum(nil)[:4])

		fmt.Printf("🔄 Processando chunk %d/%d\r", chunkIdx+1, len(chunks))

		vector, err := r.GetEmbedding(chunk)
		if err != nil {
			log.Printf("Errore embedding per chunk %s: %v", docID, err)
			continue
		}

		doc := Document{
			ID:      docID,
			Content: chunk,
			Page:    1,
			Vector:  vector,
		}

		r.vectorStore.Documents = append(r.vectorStore.Documents, doc)
		time.Sleep(50 * time.Millisecond)
	}

	r.vectorStore.ModelName = r.embedModel
	fmt.Printf("\n✅ Creati %d chunks con embedding\n", len(r.vectorStore.Documents))

	return r.SaveVectorStore()
}

// Salva vector store su file
func (r *RAGChatbot) SaveVectorStore() error {
	data, err := json.MarshalIndent(r.vectorStore, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(r.dbPath, data, 0644)
}

// Carica vector store da file
func (r *RAGChatbot) LoadVectorStore() error {
	if _, err := os.Stat(r.dbPath); os.IsNotExist(err) {
		return fmt.Errorf("database non esistente")
	}

	data, err := os.ReadFile(r.dbPath)
	if err != nil {
		return err
	}

	return json.Unmarshal(data, r.vectorStore)
}

// Ricerca documenti simili con scoring migliorato
func (r *RAGChatbot) SearchSimilar(query string, topK int) ([]Document, []float64, error) {
	queryVector, err := r.GetEmbedding(query)
	if err != nil {
		return nil, nil, err
	}

	type ScoredDocument struct {
		Document Document
		Score    float64
	}

	var scoredDocs []ScoredDocument

	for _, doc := range r.vectorStore.Documents {
		similarity := cosineSimilarity(queryVector, doc.Vector)
		scoredDocs = append(scoredDocs, ScoredDocument{
			Document: doc,
			Score:    similarity,
		})
	}

	// Ordina per similarità decrescente
	sort.Slice(scoredDocs, func(i, j int) bool {
		return scoredDocs[i].Score > scoredDocs[j].Score
	})

	// Prendi i top K
	if topK > len(scoredDocs) {
		topK = len(scoredDocs)
	}

	var results []Document
	var scores []float64
	for i := 0; i < topK; i++ {
		results = append(results, scoredDocs[i].Document)
		scores = append(scores, scoredDocs[i].Score)
	}

	return results, scores, nil
}

// Risposta ottimizzata per riportare il contenuto del manuale
func (r *RAGChatbot) GenerateManualResponse(question string, context []Document, scores []float64) (string, error) {
	// Costruisci il contesto con informazioni sui punteggi
	var contextText strings.Builder
	contextText.WriteString("CONTENUTO DEL MANUALE RELATIVO ALLA DOMANDA:\n\n")

	for i, doc := range context {
		contextText.WriteString(fmt.Sprintf("SEZIONE %d (Pagina %d, Rilevanza: %.2f):\n", i+1, doc.Page, scores[i]))
		contextText.WriteString(doc.Content)
		contextText.WriteString("\n" + strings.Repeat("-", 50) + "\n\n")
	}

	// Prompt ottimizzato per riportare contenuto del manuale
	prompt := fmt.Sprintf(`Sei un assistente che deve rispondere basandosi ESCLUSIVAMENTE sul contenuto del manuale fornito.

ISTRUZIONI:
1. Rispondi SOLO utilizzando le informazioni presenti nel contenuto del manuale
2. Riporta le informazioni del manuale in modo chiaro e organizzato
3. Se possibile, indica da quale sezione/pagina provengono le informazioni
4. Se la risposta non è presente nel manuale, dillo esplicitamente
5. NON aggiungere informazioni esterne al manuale
6. Mantieni la terminologia tecnica originale del manuale

CONTENUTO DEL MANUALE:
%s

DOMANDA: %s

RISPOSTA BASATA SUL MANUALE:`, contextText.String(), question)

	// Usa modello più ricco con parametri ottimizzati per fedeltà al testo
	reqBody := OllamaRequestAdvanced{
		Model:       r.chatModel,
		Prompt:      prompt,
		Stream:      false,
		Temperature: 0.1,  // Molto bassa per aderenza al testo
		TopK:        20,   // Limitato per coerenza
		TopP:        0.8,  // Ridotto per precisione
		NumCtx:      4096, // Contesto ampio per documenti tecnici
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	resp, err := http.Post(r.ollamaBaseURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("errore chiamata Ollama: %v", err)
	}
	defer resp.Body.Close()

	var response OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", err
	}

	return response.Response, nil
}

// Chat ottimizzata per velocità
func (r *RAGChatbot) Chat(question string) (string, []Document, []float64, error) {
	if len(r.vectorStore.Documents) == 0 {
		return "Per favore, carica prima un documento PDF o TXT.", nil, nil, nil
	}

	// Cerca meno documenti per velocità (3 invece di 5)
	similarDocs, scores, err := r.SearchSimilar(question, 3)
	if err != nil {
		return "", nil, nil, err
	}

	// Genera risposta veloce
	answer, err := r.GenerateManualResponse(question, similarDocs, scores)
	if err != nil {
		return "", nil, nil, err
	}

	return answer, similarDocs, scores, nil
}

// Verifica disponibilità modelli
func (r *RAGChatbot) CheckModelsAvailable() error {
	resp, err := http.Get(r.ollamaBaseURL + "/api/tags")
	if err != nil {
		return fmt.Errorf("Ollama non disponibile su %s: %v", r.ollamaBaseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("Ollama risponde con status: %d", resp.StatusCode)
	}

	// Verifica modelli specifici
	var tagsResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		return fmt.Errorf("errore decodifica risposta tags: %v", err)
	}

	embedFound := false
	chatFound := false

	for _, model := range tagsResp.Models {
		if strings.Contains(model.Name, r.embedModel) {
			embedFound = true
		}
		if strings.Contains(model.Name, r.chatModel) {
			chatFound = true
		}
	}

	if !embedFound {
		fmt.Printf("⚠️  Modello embedding '%s' non trovato. Installalo con: ollama pull %s\n", r.embedModel, r.embedModel)
	}
	if !chatFound {
		fmt.Printf("⚠️  Modello chat '%s' non trovato. Installalo con: ollama pull %s\n", r.chatModel, r.chatModel)
	}

	return nil
}

func main() {
	chatbot := NewRAGChatbot()

	fmt.Println("🤖 Chatbot RAG per Manuali Tecnici")
	fmt.Println("===================================")

	// Verifica Ollama e modelli
	fmt.Println("🔍 Verifica disponibilità Ollama e modelli...")
	if err := chatbot.CheckModelsAvailable(); err != nil {
		log.Fatal("❌ ", err)
	}
	fmt.Println("✅ Ollama disponibile")

	// Prova a caricare database esistente
	fmt.Println("📂 Caricamento database esistente...")
	if err := chatbot.LoadVectorStore(); err != nil {
		fmt.Println("⚠️  Nessun database esistente trovato")
	} else {
		fmt.Printf("✅ Database caricato: %d documenti (modello: %s)\n",
			len(chatbot.vectorStore.Documents), chatbot.vectorStore.ModelName)
	}

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\n📋 Opzioni disponibili:")
		fmt.Println("1. Elabora nuovo PDF")
		fmt.Println("2. Elabora nuovo TXT")
		fmt.Println("3. Configura modelli")
		fmt.Println("4. Fai una domanda")
		fmt.Println("5. Mostra statistiche database")
		fmt.Println("6. Esci")
		fmt.Print("\nScegli un'opzione (1-6): ")

		choice, _ := reader.ReadString('\n')
		choice = strings.TrimSpace(choice)

		switch choice {
		case "1":
			fmt.Print("\n📄 Inserisci il percorso del file PDF: ")
			pdfPath, _ := reader.ReadString('\n')
			pdfPath = strings.TrimSpace(pdfPath)

			if _, err := os.Stat(pdfPath); os.IsNotExist(err) {
				fmt.Println("❌ File non trovato")
				continue
			}

			fmt.Println("\n🚀 Inizio elaborazione PDF...")
			start := time.Now()

			if err := chatbot.ProcessPDF(pdfPath); err != nil {
				fmt.Printf("❌ Errore: %v\n", err)
			} else {
				duration := time.Since(start)
				fmt.Printf("✅ PDF elaborato con successo in %v\n", duration)
				fmt.Printf("📊 Documenti nel database: %d\n", len(chatbot.vectorStore.Documents))
			}

		case "2":
			fmt.Print("\n📝 Inserisci il percorso del file TXT: ")
			txtPath, _ := reader.ReadString('\n')
			txtPath = strings.TrimSpace(txtPath)

			if _, err := os.Stat(txtPath); os.IsNotExist(err) {
				fmt.Println("❌ File non trovato")
				continue
			}

			fmt.Println("\n🚀 Inizio elaborazione TXT...")
			start := time.Now()

			if err := chatbot.ProcessTXT(txtPath); err != nil {
				fmt.Printf("❌ Errore: %v\n", err)
			} else {
				duration := time.Since(start)
				fmt.Printf("✅ TXT elaborato con successo in %v\n", duration)
				fmt.Printf("📊 Documenti nel database: %d\n", len(chatbot.vectorStore.Documents))
			}

		case "3":
			fmt.Printf("\n🔧 Configurazione modelli attuali:\n")
			fmt.Printf("📝 Embedding: %s\n", chatbot.embedModel)
			fmt.Printf("💬 Chat: %s\n", chatbot.chatModel)

			fmt.Print("\n📝 Nuovo modello embedding (invio per mantenere attuale): ")
			newEmbed, _ := reader.ReadString('\n')
			newEmbed = strings.TrimSpace(newEmbed)
			if newEmbed != "" {
				chatbot.embedModel = newEmbed
			}

			fmt.Print("💬 Nuovo modello chat (invio per mantenere attuale): ")
			newChat, _ := reader.ReadString('\n')
			newChat = strings.TrimSpace(newChat)
			if newChat != "" {
				chatbot.chatModel = newChat
			}

			chatbot.SetModels(chatbot.embedModel, chatbot.chatModel)

		case "4":
			if len(chatbot.vectorStore.Documents) == 0 {
				fmt.Println("⚠️  Carica prima un documento!")
				continue
			}

			fmt.Print("\n❓ Inserisci la tua domanda: ")
			question, _ := reader.ReadString('\n')
			question = strings.TrimSpace(question)

			if question == "" {
				continue
			}

			fmt.Println("\n🤔 Analizzando il manuale...")
			start := time.Now()

			answer, sources, scores, err := chatbot.Chat(question)
			if err != nil {
				fmt.Printf("❌ Errore: %v\n", err)
				continue
			}

			duration := time.Since(start)
			fmt.Printf("\n💬 Risposta dal manuale (generata in %v):\n", duration)
			fmt.Printf("═══════════════════════════════════════════════════════\n")
			fmt.Println(answer)

			if len(sources) > 0 {
				fmt.Println("\n📚 Sezioni del manuale consultate:")
				for i, source := range sources {
					fmt.Printf("\n📍 Sezione %d (Pagina %d, Rilevanza: %.2f):\n", i+1, source.Page, scores[i])
					preview := source.Content
					if len(preview) > 200 {
						preview = preview[:200] + "..."
					}
					fmt.Printf("   %s\n", preview)
				}
			}

		case "5":
			fmt.Printf("\n📊 Statistiche Database:\n")
			fmt.Printf("═══════════════════════════\n")
			fmt.Printf("📄 Documenti totali: %d\n", len(chatbot.vectorStore.Documents))
			fmt.Printf("📝 Modello embedding: %s\n", chatbot.vectorStore.ModelName)
			fmt.Printf("💬 Modello chat: %s\n", chatbot.chatModel)

			if len(chatbot.vectorStore.Documents) > 0 {
				pageCount := make(map[int]int)
				totalChars := 0
				for _, doc := range chatbot.vectorStore.Documents {
					pageCount[doc.Page]++
					totalChars += len(doc.Content)
				}

				fmt.Printf("📖 Pagine elaborate: %d\n", len(pageCount))
				fmt.Printf("📊 Caratteri totali: %d\n", totalChars)
				fmt.Printf("📏 Media caratteri per chunk: %.0f\n", float64(totalChars)/float64(len(chatbot.vectorStore.Documents)))
			}

		case "6":
			fmt.Println("\n👋 Arrivederci!")
			return

		default:
			fmt.Println("❌ Opzione non valida")
		}
	}
}
