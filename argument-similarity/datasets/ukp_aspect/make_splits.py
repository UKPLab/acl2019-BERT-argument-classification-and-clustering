"""
This script takes the UKP Argument Aspect Similarity Corpus (https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/ukp_argument_aspect_similarity_corpus/ukp_argument_aspect_similarity_corpus.en.jsp) and creates a 4-fold cross-topic split

To run this script, download the corpus, unzip it so that this folder contains the UKP_ASPECT.tsv file
"""


import os

topic_splits = [
    {
'train': ['Wind power', 'Nanotechnology', '3d printing', 'Cryptocurrency', 'Virtual reality', 'Gene editing', 'Public surveillance', 'Genetic diagnosis', 'Geoengineering', 'Gmo', 'Organ donation', 'Recycling', 'Offshore drilling', 'Robotic surgery', 'Cloud storing', 'Electric cars', 'Stem cell research'],
'dev': ['Hydrogen fuel cells', 'Electronic voting', 'Drones', 'Solar energy'],
'test': ['Tissue engineering', 'Big data', 'Fracking', 'Social networks', 'Net neutrality', 'Hydroelectric dams', 'Internet of things']
    },
    {
'train': ['Wind power', '3d printing', 'Cryptocurrency', 'Tissue engineering', 'Gene editing', 'Virtual reality', 'Big data', 'Fracking', 'Public surveillance', 'Genetic diagnosis', 'Hydroelectric dams', 'Drones', 'Gmo', 'Organ donation', 'Solar energy', 'Electronic voting', 'Electric cars'],
'dev': ['Net neutrality', 'Internet of things', 'Hydrogen fuel cells', 'Social networks'],
'test': ['Nanotechnology', 'Geoengineering', 'Recycling', 'Offshore drilling', 'Robotic surgery', 'Cloud storing', 'Stem cell research']
    },
    {
'train': ['Nanotechnology', 'Tissue engineering', 'Hydrogen fuel cells', 'Big data', 'Cloud storing', 'Fracking', 'Net neutrality', 'Hydroelectric dams', 'Geoengineering', 'Gmo', 'Recycling', 'Offshore drilling', 'Robotic surgery', 'Internet of things', 'Electronic voting', 'Genetic diagnosis', 'Stem cell research'],
'dev': ['Gene editing', 'Solar energy', 'Social networks', '3d printing'],
'test': ['Wind power', 'Cryptocurrency', 'Virtual reality', 'Public surveillance', 'Drones', 'Organ donation', 'Electric cars']
    },
    {
'train': ['Wind power', 'Nanotechnology', 'Tissue engineering', 'Virtual reality', 'Big data', 'Fracking', 'Public surveillance', 'Social networks', 'Net neutrality', 'Drones', 'Recycling', 'Organ donation', 'Offshore drilling', 'Robotic surgery', 'Cloud storing', 'Electric cars', 'Stem cell research'],
'dev': ['Cryptocurrency', 'Hydroelectric dams', 'Geoengineering', 'Internet of things'],
'test': ['3d printing', 'Gene editing', 'Hydrogen fuel cells', 'Gmo', 'Solar energy', 'Electronic voting', 'Genetic diagnosis']
    }
]

sentences = {}

with open('UKP_ASPECT.tsv') as fIn:
    next(fIn) #Skip header line
    for line in fIn:
        line = line.strip()
        topic = line.split('\t')[0]
        if topic not in sentences:
            sentences[topic] = []

        sentences[topic].append(line)


for split_idx in range(len(topic_splits)):
    for dataset_split in ['train', 'dev', 'test']:
        folder = os.path.join("splits", str(split_idx))
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, dataset_split+'.tsv'), 'w') as fOut:
            for topic in topic_splits[split_idx][dataset_split]:
                for sentence in sentences[topic]:
                    fOut.write(sentence+"\n")
            fOut.flush()

print("Splits created")

##Create all_data.tsv.gz
with open(os.path.join('splits', 'all_data.tsv'), 'w') as fOut:
    for topic in sentences:
        for sentence in sentences[topic]:
            fOut.write(sentence + "\n")

print("all_data.tsv.gz created")
print("topics:", sentences.keys())