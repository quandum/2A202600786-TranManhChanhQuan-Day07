# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Mạnh Chánh Quân
Mã học viên: 2A202600786
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

> Hai văn bản có cosine similarity cao nghĩa là vector embedding của chúng trỏ gần cùng một hướng trong không gian vector, tức là chúng chia sẻ ý nghĩa ngữ nghĩa tương tự nhau. Giá trị gần 1.0 có nghĩa là gần như giống nhau về mặt ngữ nghĩa, trong khi 0 có nghĩa là không liên quan, và -1 có nghĩa là trái nghĩa hoàn toàn.

**Ví dụ HIGH similarity:**

- Sentence A: "The dog ran quickly across the park."
- Sentence B: "A canine sprinted through the garden."
- Tại sao tương đồng: Cả hai câu đều mô tả một con chó đang chạy trong một không gian xanh, dùng từ đồng nghĩa (dog/canine, ran/sprinted, park/garden) nên embedding sẽ rất gần nhau.

**Ví dụ LOW similarity:**

- Sentence A: "The stock market closed at a record high today."
- Sentence B: "I enjoy eating chocolate ice cream in summer."
- Tại sao khác: Hai câu thuộc hoàn toàn hai domain khác nhau (tài chính vs. thực phẩm/sở thích cá nhân), không có từ hoặc khái niệm chung, nên vector embedding sẽ trỏ theo hai hướng khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

> Cosine similarity đo góc giữa hai vector chứ không phải khoảng cách tuyến tính, nên nó không bị ảnh hưởng bởi độ dài (norm) của vector — văn bản dài hay ngắn có thể vẫn có cosine similarity cao nếu chúng nói về cùng chủ đề. Ngược lại, Euclidean distance bị ảnh hưởng bởi magnitude, dẫn đến văn bản dài luôn bị coi là "xa" hơn dù cùng ý nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> Áp dụng công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
>
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = **23 chunks**`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

> Với overlap=100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = **25 chunks**`
> Chunk count tăng lên vì step nhỏ hơn (mỗi bước chỉ tiến 400 ký tự thay vì 450). Overlap lớn hơn giúp bảo toàn ngữ cảnh ở ranh giới giữa các chunk — câu hoặc ý tưởng quan trọng sẽ không bị cắt ngang và bị mất, giúp retrieval chính xác hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI/RAG Systems & Programming Knowledge Base

**Tại sao nhóm chọn domain này?**

> Domain AI/RAG là phù hợp nhất cho lab này vì tài liệu đã có sẵn và phong phú trong thư mục `data/`. Các tài liệu bao gồm thiết kế hệ thống RAG, kỹ thuật retrieval, và hướng dẫn lập trình — cho phép kiểm tra các query đa dạng từ kỹ thuật đến hỗ trợ người dùng. Domain này cũng có cả tài liệu tiếng Anh lẫn tiếng Việt, giúp thử nghiệm metadata filtering theo ngôn ngữ.

### Data Inventory

| # | Tên tài liệu               | Nguồn | Số ký tự | Metadata đã gán            |
| - | ----------------------------- | ------ | ----------- | ----------------------------- |
| 1 | customer_support_playbook.txt | data/  | 1,703       | category=support, lang=en     |
| 2 | python_intro.txt              | data/  | 1,953       | category=programming, lang=en |
| 3 | rag_system_design.md          | data/  | 2,416       | category=ai, lang=en          |
| 4 | vector_store_notes.md         | data/  | 2,149       | category=ai, lang=en          |
| 5 | vi_retrieval_notes.md         | data/  | 2,188       | category=ai, lang=vi          |

### Metadata Schema

| Trường metadata | Kiểu  | Ví dụ giá trị                          | Tại sao hữu ích cho retrieval?                                                                 |
| ----------------- | ------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `category`      | string | `"ai"`, `"support"`, `"programming"` | Cho phép filter theo chủ đề — query về lập trình không cần xét tài liệu support      |
| `lang`          | string | `"en"`, `"vi"`                         | Cho phép filter theo ngôn ngữ — query tiếng Việt có thể ưu tiên tài liệu tiếng Việt |
| `source`        | string | `"rag_system_design.md"`                 | Truy vết nguồn gốc chunk để kiểm tra và debug retrieval                                    |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên `python_intro.txt` (chunk_size=200):

| Tài liệu       | Strategy                           | Chunk Count | Avg Length | Preserves Context?                       |
| ---------------- | ---------------------------------- | ----------- | ---------- | ---------------------------------------- |
| python_intro.txt | FixedSizeChunker (`fixed_size`)  | 13          | 195.7      | Trung bình — cắt ngang câu           |
| python_intro.txt | SentenceChunker (`by_sentences`) | 5           | 387.0      | Tốt — giữ nguyên câu hoàn chỉnh   |
| python_intro.txt | RecursiveChunker (`recursive`)   | 18          | 108.0      | Tốt — tôn trọng cấu trúc văn bản |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=300)

**Mô tả cách hoạt động:**

> `RecursiveChunker` thử lần lượt các separator theo thứ tự ưu tiên: `"\n\n"` (đoạn văn), `"\n"` (dòng), `". "` (câu), `" "` (từ), `""` (ký tự). Nó dùng separator phù hợp nhất để tách văn bản, sau đó gom các phần nhỏ lại đến khi gần đạt `chunk_size`. Nếu một phần vẫn còn quá lớn, nó đệ quy với separator tiếp theo. Kết quả là các chunk tôn trọng ranh giới tự nhiên của văn bản.

**Tại sao tôi chọn strategy này cho domain nhóm?**

> Domain AI/RAG documents có cấu trúc Markdown với headers và đoạn văn rõ ràng. `RecursiveChunker` phân tách theo `"\n\n"` trước (giữ nguyên đoạn văn hoàn chỉnh), sau đó mới cắt sâu hơn nếu cần — phù hợp hơn `FixedSizeChunker` (hay cắt giữa câu) và `SentenceChunker` (tạo chunk quá dài khi đoạn văn nhiều câu).

**Code snippet:**

```python
chunker = RecursiveChunker(chunk_size=300)
chunks = chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu       | Strategy                               | Chunk Count | Avg Length | Retrieval Quality?                      |
| ---------------- | -------------------------------------- | ----------- | ---------- | --------------------------------------- |
| python_intro.txt | FixedSizeChunker (baseline)            | 13          | 195.7      | Thấp — cắt ngang câu                |
| python_intro.txt | **RecursiveChunker (của tôi)** | 18          | 108.0      | Cao — chunk nhỏ hơn, tập trung hơn |

### So Sánh Với Thành Viên Khác

| Thành viên       | Strategy                                          | Retrieval Score (/10) | Điểm mạnh                                                     | Điểm yếu                                                               |
| ------------------ | ------------------------------------------------- | --------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Tôi               | RecursiveChunker(300)                             | 7                     | Tôn trọng cấu trúc văn bản; chunk nhỏ, tập trung           | Chunk kích thước không đều; kém hiệu quả với MockEmbedder             |
| Hồ Thành Tiến   | HybridChunker (heading → semantic → fixed-size) | 8                     | Giữ cấu trúc markdown tốt; chunk coherent trên file `.md` | Không lợi thế trên `.txt`; mock embed làm retrieval yếu ở Q1, Q3 |
| Nguyễn Thái Bảo | SentenceChunker (avg length 366.0)                | 10                    | Mỗi chunk là câu hoàn chỉnh; ranh giới ngữ nghĩa rõ ràng  | Chunk dài khi đoạn văn nhiều câu; kém linh hoạt với văn bản phi cấu trúc |

**Strategy nào tốt nhất cho domain này? Tại sao?**

> Trong thực nghiệm này, `SentenceChunker` đạt điểm cao nhất (10/10) vì các tài liệu trong corpus có câu văn rõ ràng, mỗi câu thường mang một ý nghĩa độc lập — phù hợp với MockEmbedder vốn nhạy với ranh giới ký tự hơn là ngữ nghĩa. `HybridChunker` (8/10) của Hồ Thành Tiến cho thấy ưu thế rõ rệt trên các file `.md` có heading phân cấp, nhưng giảm hiệu quả trên `.txt` thuần. `RecursiveChunker` (7/10) của tôi tạo chunk nhỏ và tôn trọng cấu trúc văn bản, nhưng chunk không đều về kích thước có thể làm nhiễu ranking với MockEmbedder. Nếu dùng real semantic embeddings, `RecursiveChunker` hoặc `HybridChunker` sẽ có lợi thế hơn nhờ bảo toàn ngữ cảnh tốt hơn.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

> Dùng `re.split(r'(?<=[.!?]) |(?<=\.)\n', text)` để phát hiện ranh giới câu dựa trên dấu `.`, `!`, `?` theo sau bởi khoảng trắng hoặc newline (positive lookbehind). Sau khi tách câu, các câu được gom lại theo nhóm `max_sentences_per_chunk` câu và join bằng dấu cách. Edge case xử lý: câu rỗng được lọc bỏ, strip whitespace, và `max_sentences_per_chunk` tối thiểu là 1.

**`RecursiveChunker.chunk` / `_split`** — approach:

> Algorithm đệ quy với base case là `len(text) <= chunk_size` — trả về nguyên văn bản. Nếu văn bản quá lớn, thử tìm separator hiện tại; nếu không có, đệ quy với separator tiếp theo. Khi tách được, dùng buffer để gom các phần nhỏ cho đến khi tổng vượt `chunk_size`, rồi flush buffer và tiếp tục. Phần quá lớn được đệ quy xử lý tiếp với các separator nhỏ hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:

> `add_documents` embed content của mỗi `Document` qua `_embedding_fn`, đóng gói cùng metadata thành dict, và lưu vào `self._store` (in-memory) hoặc ChromaDB collection. `search` embed query rồi tính dot product với tất cả stored embeddings (vì vectors đã được normalize, dot product ≡ cosine similarity), sắp xếp giảm dần theo score, và trả về top_k kết quả.

**`search_with_filter` + `delete_document`** — approach:

> `search_with_filter` filter trước: lọc `self._store` chỉ giữ records có metadata khớp với tất cả key-value trong `metadata_filter`, sau đó gọi `_search_records` trên tập đã lọc. `delete_document` lọc bỏ tất cả records có `metadata["doc_id"] == doc_id` và trả về `True` nếu count giảm.

### KnowledgeBaseAgent

**`answer`** — approach:

> Lấy `top_k` chunks liên quan nhất từ store qua `store.search(question)`. Nối các chunks bằng `"\n\n"` làm context. Build prompt theo pattern RAG chuẩn: `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"`. Truyền prompt vào `llm_fn` và trả về kết quả. Cấu trúc prompt đơn giản nhưng hiệu quả, đặt context trước question để LLM có thể ground câu trả lời.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.14.5, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\quand\OneDrive\Documents\VinUni-AI2k_2\Day07 lab

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================= 42 passed in 0.14s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

> **Lưu ý:** Scores dưới đây dùng `MockEmbedder` (hash-based, không semantic). Với real embeddings (sentence-transformers), kết quả sẽ phản ánh đúng nghĩa ngữ nghĩa hơn.

| Pair | Sentence A                             | Sentence B                                     | Dự đoán | Actual Score | Đúng? |
| ---- | -------------------------------------- | ---------------------------------------------- | ---------- | ------------ | ------- |
| 1    | The cat sat on the mat.                | A kitten rested on the rug.                    | high       | 0.0578       | ❌      |
| 2    | Python is a programming language.      | Python is used in data science.                | high       | 0.1156       | ❌      |
| 3    | I love eating pizza.                   | The stock market crashed today.                | low        | -0.0517      | ✅      |
| 4    | Machine learning uses neural networks. | Deep learning is a subset of machine learning. | high       | -0.1842      | ❌      |
| 5    | The weather is sunny today.            | Quantum mechanics governs subatomic particles. | low        | 0.1394       | ❌      |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**

> Bất ngờ nhất là cặp 4: "Machine learning uses neural networks" và "Deep learning is a subset of machine learning" — hai câu rõ ràng liên quan chặt chẽ nhưng lại có score âm (-0.18). Điều này phản ánh hạn chế của `MockEmbedder` dùng MD5 hash — nó không học được ý nghĩa ngữ nghĩa mà chỉ tạo vector ngẫu nhiên dựa trên chuỗi ký tự. Với real semantic embeddings, cặp này sẽ có score > 0.7.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query                                           | Gold Answer                                                                                                                                                                               |
| - | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | What is RAG and how does it work?               | RAG (Retrieval-Augmented Generation) retrieves relevant chunks from a vector store, then passes them as context to an LLM to generate a grounded answer.                                  |
| 2 | How to handle customer complaints?              | Acknowledge the issue, retrieve relevant knowledge base articles, provide a clear resolution, and escalate if needed.                                                                     |
| 3 | What are Python data types?                     | Python has built-in data types including int, float, str, list, tuple, dict, set, and bool.                                                                                               |
| 4 | How does vector similarity search work?         | Vector similarity search embeds both the query and documents into a shared vector space, then ranks documents by cosine similarity (or dot product) to the query vector.                  |
| 5 | What is chunking strategy in retrieval systems? | Chunking splits documents into smaller pieces before embedding; strategies include fixed-size, sentence-based, and recursive splitting — each affecting retrieval precision differently. |

### Kết Quả Của Tôi

Sử dụng `RecursiveChunker(chunk_size=300)` + `EmbeddingStore` + `MockEmbedder`, 73 chunks tổng cộng.

| # | Query                                           | Top-1 Retrieved Chunk (tóm tắt)                                                                            | Score  | Relevant?  | Agent Answer (tóm tắt)      |
| - | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------ | ---------- | ----------------------------- |
| 1 | What is RAG and how does it work?               | "For example, a support assistant might restrict retrieval..." (vector_store_notes.md)                       | 0.3305 | Một phần | [LLM answer based on context] |
| 2 | How to handle customer complaints?              | "The retrieval layer embeds user questions, performs similarity search..." (rag_system_design.md)            | 0.3029 | Không     | [LLM answer based on context] |
| 3 | What are Python data types?                     | "A common vector search pipeline has four stages..." (vector_store_notes.md)                                 | 0.2154 | Không     | [LLM answer based on context] |
| 4 | How does vector similarity search work?         | "In practice, honest uncertainty is better than a polished but incorrect..." (customer_support_playbook.txt) | 0.3555 | Không     | [LLM answer based on context] |
| 5 | What is chunking strategy in retrieval systems? | "That is why teams should test retrieval quality with realistic queries..." (vector_store_notes.md)          | 0.2356 | Một phần | [LLM answer based on context] |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

> **Phân tích:** MockEmbedder không phản ánh semantic similarity thực, dẫn đến retrieval kém. Với `sentence-transformers` hoặc OpenAI embeddings thực, kết quả sẽ tốt hơn rõ rệt.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

> Thành viên khác dùng `SentenceChunker` và cho thấy rằng với các tài liệu dạng Q&A ngắn, mỗi câu thực sự là một đơn vị thông tin độc lập — trong trường hợp đó SentenceChunker vượt trội hơn RecursiveChunker vì tạo ra chunk nhỏ hơn và focused hơn. Điều này dạy tôi rằng không có strategy nào tốt nhất cho mọi loại tài liệu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

> Nhóm khác minh họa cách dùng metadata filtering hiệu quả để tăng precision: thay vì search toàn bộ corpus, lọc theo `lang="vi"` trước giúp loại bỏ ngay các chunk không liên quan về ngôn ngữ. Đây là kỹ thuật quan trọng khi corpus đa ngôn ngữ.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

> Tôi sẽ dùng real embeddings (`sentence-transformers/all-MiniLM-L6-v2`) thay vì MockEmbedder để kết quả retrieval thực sự phản ánh ngữ nghĩa. Ngoài ra, tôi sẽ thêm nhiều trường metadata hơn (ví dụ: `topic`, `difficulty`) và thiết kế queries yêu cầu metadata filtering để kiểm tra end-to-end từ chunking đến retrieval.

**Failure Case (Ex 3.5):**

> Query "What are Python data types?" trả về chunks về vector search thay vì nội dung Python — lý do là MockEmbedder hash-based không nắm được semantic. Hơn nữa, `python_intro.txt` có ít chunk liên quan, và chunk quan trọng nhất ("Python has built-in types: int, float, str...") bị chôn vùi sâu trong file. Giải pháp: dùng real embeddings + thêm metadata `category="programming"` và filter theo đó khi cần.

---

## Tự Đánh Giá

| Tiêu chí                  | Loại     | Điểm tự đánh giá |
| --------------------------- | --------- | ---------------------- |
| Warm-up                     | Cá nhân | 5 / 5                  |
| Document selection          | Nhóm     | 9 / 10                 |
| Chunking strategy           | Nhóm     | 13 / 15                |
| My approach                 | Cá nhân | 10 / 10                |
| Similarity predictions      | Cá nhân | 4 / 5                  |
| Results                     | Cá nhân | 8 / 10                 |
| Core implementation (tests) | Cá nhân | 30 / 30                |
| Demo                        | Nhóm     | 4 / 5                  |
| **Tổng**             |           | **83 / 100**     |
