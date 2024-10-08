from gliner import GLiNER, GLiNERConfig
from gliner.decoding import SpanLinkerDecoder
from gliner.gliner_db import GLiNERDBConfig, AutoGLiNERDb, HNSW
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


model = GLiNER.from_pretrained("knowledgator/gliner-bi-large-v1.0")
model.config.entity_linking = "span_linking"
model.decoder = SpanLinkerDecoder(model.config)

model.to("cuda")

db = AutoGLiNERDb.from_pretrained("BioMike/PubMedMeShHNSWfaiss")

text = """"
The BRCA1 gene, which codes for a protein essential to DNA repair, acts as a guardian of genomic integrity. When mutations arise in the BRCA1 gene, the risk of developing breast cancer and ovarian cancer surges, as these common forms of cancer result from the compromised ability to fix DNA damage. The faulty BRCA1 protein allows DNA errors to accumulate, ultimately setting the stage for disease development. Recent research also spotlights p53, a powerful protein known as the "guardian of the genome," which plays a pivotal role in regulating the cell cycle. Mutations in the p53 gene are prevalent across a wide range of cancers, from lung cancer to colorectal cancer. Understanding the interaction between p53 and BRCA1 pathways is proving crucial in cancer research, unlocking potential new avenues for therapy. In a different realm of disease, inflammatory bowel disease (IBD), which includes Crohn’s disease and ulcerative colitis, exemplifies a chronic condition driven by both genetic predisposition and environmental triggers. Aberrant immune responses and imbalanced expression of key proteins like TNF-alpha and interleukin-6 have been identified as major players in the progression of IBD. These findings continue to fuel research into understanding and combating these complex diseases.
"""
labels = [ "protein", "disease", "gene", "pathway", "biological process"]

# text = """"The lion, often referred to as the king of the jungle, is a large carnivorous feline known for its strength and majestic presence. The elephant, the largest land animal, uses its trunk for grasping objects and is famous for its intelligence and memory. In the skies, the eagle soars high, a bird of prey with keen vision and powerful flight, while the hawk is another skilled predator with sharp eyesight and hunting abilities.

# """
# labels = [ "animal"]

entities = model.link_entities(text, labels, vector_db=db, threshold=0.01)
print(entities)

for entity in entities:
    for i in range(5):
        print(f'{entity["label"]}: ->: {entity["text"]}: linked to {i+1}: {db.ontology[str(entity["linked_entity"]["ids"][i])]["label"]}')