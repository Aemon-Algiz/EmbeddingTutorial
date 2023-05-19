from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-xl')
sentence = "I like dinosaurs and I want to see a movie about them"
instruction = "A scifi movie title:"
embeddings = model.encode([[instruction, sentence]])
print(embeddings)