# IPTC Media Topic Classification

This repository documents the development of the IPTC Media Topic classifier that provides single-label classification using the 17 top-level topic labels from the [IPTC NewsCodes Media Topic](https://www.iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html) hierarchical schema.

## Requirements

Before starting the project make sure these requirements are available:

- [python]. For setting up the environment and Python dependencies (version 3.11.10 or higher).

### Setup

Recreating the environment that was used for the experiments, using conda:

Create the environment:
```bash
conda create -n IPTC_env python=3.11.10
```

Add the requirements:
```bash
conda env update -n IPTC_env -f environment.yml --prune
```

Activate the new environment:
```bash
conda activate IPTC_env
```

## IPTC NewsCodes Media Topic schema

Since 2010, the International Press Telecommunications Council (IPTC) maintains a taxonomy for the categorization of news text. This taxonomy takes the form of a tree with 17 top-level topics such as politics, society, or sport. Each topic branches into subtopics until very specific topics are reached, such as adult education, impeachment, or missing person. The taxonomy can be visualized at https://show.newscodes.org/index.html?newscodes=medtop&lang=en-GB&startTo=Show.

For more information on the labels and the schema, see the [IPTC NewsCode guidelines](https://iptc.org/std/NewsCodes/guidelines/).

Information on all labels, their levels, parent and child labels and definitions can be accessed from the [original spreadsheet](datasets/IPTC-MediaTopic-NewsCodes-mappings.xlsx) or the extracted JSON dictionary (datasets/iptc_mapping.json)[datasets/iptc_mapping.json]. We use the version of the schema from October 24, 2023.

```
labels = ['disaster, accident and emergency incident',
 'human interest',
 'politics',
 'education',
 'crime, law and justice',
 'economy, business and finance',
 'conflict, war and peace',
 'arts, culture, entertainment and media',
 'labour',
 'weather',
 'religion',
 'society',
 'health',
 'environment',
 'lifestyle and leisure',
 'science and technology',
 'sport']
```

Description of the 17 top labels:
```
Extended description

Based on the description of the labels and sub-labels here: https://www.iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html

label_dict_with_description_ext = {
	'disaster, accident and emergency incident - man-made or natural events resulting in injuries, death or damage, e.g., explosions, transport accidents, famine, drowning, natural disasters, emergency planning and response.': 0,
	'human interest - news about life and behavior of royalty and celebrities, news about obtaining awards, ceremonies (graduation, wedding, funeral, celebration of launching something), birthdays and anniversaries, and news about silly or stupid human errors.': 1,
	'politics - news about local, regional, national and international exercise of power, including news about election, fundamental rights, government, non-governmental organisations, political crises, non-violent international relations, public employees, government policies.': 2,
	'education - all aspects of furthering knowledge, formally or informally, including news about schools, curricula, grading, remote learning, teachers and students.': 3,
	'crime, law and justice - news about committed crime and illegal activities, the system of courts, law and law enforcement (e.g., judges, lawyers, trials, punishments of offenders).': 4,
	'economy, business and finance - news about companies, products and services, any kind of industries, national economy, international trading, banks, (crypto)currency, business and trade societies, economic trends and indicators (inflation, employment statistics, GDP, mortgages, ...), international economic institutions, utilities (electricity, heating, waste management, water supply).': 5,
	'conflict, war and peace - news about terrorism, wars, wars victims, cyber warfare, civil unrest (demonstrations, riots, rebellions), peace talks and other peace activities.': 6,
	'arts, culture, entertainment and media - news about cinema, dance, fashion, hairstyle, jewellery, festivals, literature, music, theatre, TV shows, painting, photography, woodworking, art exhibitions, libraries and museums, language, cultural heritage, news media, radio and television, social media, influencers, and disinformation.': 7,
	'labour - news about employment, employment legislation, employees and employers, commuting, parental leave, volunteering, wages, social security, labour market, retirement, unemployment, unions.': 8,
	'weather - news about weather forecasts, weather phenomena and weather warning.': 9,
	'religion - news about religions, cults, religious conflicts, relations between religion and government, churches, religious holidays and festivals, religious leaders and rituals, and religious texts.': 10,
	'society - news about social interactions (e.g., networking), demographic analyses, population census, discrimination, efforts for inclusion and equity, emigration and immigration, communities of people and minorities (LGBTQ, older people, children, indigenous people, etc.), homelessness, poverty, societal problems (addictions, bullying), ethical issues (suicide, euthanasia, sexual behavior) and social services and charity, relationships (dating, divorce, marriage), family (family planning, adoption, abortion, contraception, pregnancy, parenting).': 11,
	'health - news about diseases, injuries, mental health problems, health treatments, diets, vaccines, drugs, government health care, hospitals, medical staff, health insurance.': 12,
	'environment - news about climate change, energy saving, sustainability, pollution, population growth, natural resources, forests, mountains, bodies of water, ecosystem, animals, flowers and plants.': 13,
	'lifestyle and leisure - news about hobbies, clubs and societies, games, lottery, enthusiasm about food or drinks, car/motorcycle lovers, public holidays, leisure venues (amusement parks, cafes, bars, restaurants, etc.), exercise and fitness, outdoor recreational activities (e.g., fishing, hunting), travel and tourism, mental well-being, parties, maintaining and decorating house and garden.': 14,
	'science and technology -  news about natural sciences and social sciences, mathematics, technology and engineering, scientific institutions, scientific research, scientific publications and innovation.': 15,
	'sport - news about sports that can be executed in competitions - basketball, football, swimming, athletics, chess, dog racing, diving, golf, gymnastics, martial arts, climbing, etc.; sport achievements, sport events, sport organisation, sport venues (stadiums, gymnasiums, ...), referees, coaches, sport clubs, drug use in sport.': 16}
```

Additionally, for the manual annotation, we implemented 3 additional labels to mark the text that should be discarded (due to being unsuitable or too ambigious - see [Annotation Guidelines](IPTC_Annotation_Guidelines.pdf) for the description of the labels):
``` ["do not know", "not news", "multiple"]```


## Data

Final training and test datasets are available at:
- training dataset, annotated with GPT-4o: *EMMediaTopic dataset*: published on [the CLARIN.SI repository](http://hdl.handle.net/11356/1991)
- test dataset, manually annotated: *IPTC-top-test.jsonl* - available upon request to the authors (kuzman.taja at ijs.si) via [private GitHub repository](https://github.com/clarinsi/IPTC-top-test) inside the CLARIN.SI group

## Data Development

We took samples from the Croatian, Slovenian, Catalan and Greek monolingual [MaCoCu-Genre](http://hdl.handle.net/11356/1969) (Kuzman & Ljubešić, 2024) corpora. More precisely, we took texts that:
- were annotated as "News" with the genre classifier
- were truncated to 512 words (the limitation of BERT-like classifiers)
- consist only of the target language (no paragraphs in the foreign language)
- do not consist of multiple shortened texts - we removed all texts that consist of more than one (...) which is typical for pages with text summaries, continuing with "Read more ..."

According to these criteria, we extracted a random sample of 5,250 texts per language for training and development splits and 2,000 texts per language for the test split.

Code: [data-development-code/0-extract_sample_from_MaCoCu_10000_instances.ipynb](data-development-code/0-extract_sample_from_MaCoCu_10000_instances.ipynb)

### Automatic Annotation with GPT-4o

Code: [data-development-code/1-annotate-with-gpt4o-and-split-training-samples.ipynb](data-development-code/1-annotate-with-gpt4o-and-split-training-samples.ipynb)

We used the GPT-4o model ("gpt-4o-2024-05-13") and the following prompt:

```python
label_dict_with_description_ext = {
	'disaster, accident and emergency incident - man-made or natural events resulting in injuries, death or damage, e.g., explosions, transport accidents, famine, drowning, natural disasters, emergency planning and response.': 0,
	'human interest - news about life and behavior of royalty and celebrities, news about obtaining awards, ceremonies (graduation, wedding, funeral, celebration of launching something), birthdays and anniversaries, and news about silly or stupid human errors.': 1,
	'politics - news about local, regional, national and international exercise of power, including news about election, fundamental rights, government, non-governmental organisations, political crises, non-violent international relations, public employees, government policies.': 2,
	'education - all aspects of furthering knowledge, formally or informally, including news about schools, curricula, grading, remote learning, teachers and students.': 3,
	'crime, law and justice - news about committed crime and illegal activities, the system of courts, law and law enforcement (e.g., judges, lawyers, trials, punishments of offenders).': 4,
	'economy, business and finance - news about companies, products and services, any kind of industries, national economy, international trading, banks, (crypto)currency, business and trade societies, economic trends and indicators (inflation, employment statistics, GDP, mortgages, ...), international economic institutions, utilities (electricity, heating, waste management, water supply).': 5,
	'conflict, war and peace - news about terrorism, wars, wars victims, cyber warfare, civil unrest (demonstrations, riots, rebellions), peace talks and other peace activities.': 6,
	'arts, culture, entertainment and media - news about cinema, dance, fashion, hairstyle, jewellery, festivals, literature, music, theatre, TV shows, painting, photography, woodworking, art exhibitions, libraries and museums, language, cultural heritage, news media, radio and television, social media, influencers, and disinformation.': 7,
	'labour - news about employment, employment legislation, employees and employers, commuting, parental leave, volunteering, wages, social security, labour market, retirement, unemployment, unions.': 8,
	'weather - news about weather forecasts, weather phenomena and weather warning.': 9,
	'religion - news about religions, cults, religious conflicts, relations between religion and government, churches, religious holidays and festivals, religious leaders and rituals, and religious texts.': 10,
	'society - news about social interactions (e.g., networking), demographic analyses, population census, discrimination, efforts for inclusion and equity, emigration and immigration, communities of people and minorities (LGBTQ, older people, children, indigenous people, etc.), homelessness, poverty, societal problems (addictions, bullying), ethical issues (suicide, euthanasia, sexual behavior) and social services and charity, relationships (dating, divorce, marriage), family (family planning, adoption, abortion, contraception, pregnancy, parenting).': 11,
	'health - news about diseases, injuries, mental health problems, health treatments, diets, vaccines, drugs, government health care, hospitals, medical staff, health insurance.': 12,
	'environment - news about climate change, energy saving, sustainability, pollution, population growth, natural resources, forests, mountains, bodies of water, ecosystem, animals, flowers and plants.': 13,
	'lifestyle and leisure - news about hobbies, clubs and societies, games, lottery, enthusiasm about food or drinks, car/motorcycle lovers, public holidays, leisure venues (amusement parks, cafes, bars, restaurants, etc.), exercise and fitness, outdoor recreational activities (e.g., fishing, hunting), travel and tourism, mental well-being, parties, maintaining and decorating house and garden.': 14,
	'science and technology -  news about natural sciences and social sciences, mathematics, technology and engineering, scientific institutions, scientific research, scientific publications and innovation.': 15,
	'sport - news about sports that can be executed in competitions - basketball, football, swimming, athletics, chess, dog racing, diving, golf, gymnastics, martial arts, climbing, etc.; sport achievements, sport events, sport organisation, sport venues (stadiums, gymnasiums, ...), referees, coaches, sport clubs, drug use in sport.': 16}

	structured_prompt_label_description = f"""
			### Task
			Your task is to classify the provided text into a topic label, meaning that you need to recognize what is the topic of the text. You will be provided with a news text, delimited by single quotation marks. Always provide a label, even if you are not sure.

			### Output format
			Return a valid JSON dictionary with the following key: 'topic' and a value should be an integer which represents one of the labels according to the following dictionary: {label_dict_with_description_ext}.
			"""

```

Prediction on 2000 instances took 25 minutes and cost 15€ for each language.

Label distribution on automatically-annotated training and development data:

![](figures/MaCoCu-main-20k-sample-label-distribution.png)

### Manual Annotation

The guidelines for manual annotation based on the 17 top-level IPTC NewsCodes Media Topic labels are available [here](data/IPTC_Annotation_Guidelines.pdf). The descriptions were developed by the authors of this research to provide more details about which subcategories are included in the top-level labels.

Manual annotation was performed by 1 annotator. For inter-annotator agreement evaluation, additional annotator was used.

Out of the sample for the test dataset (2,000 texts per language) that was automatically annotated by GPT-4o, we extracted smaller samples to be manually annotated. The samples are balanced across labels and consist of 18 instances per label (if possible - some labels had less instances in the sample of 2,000 texts) -> around 300 instances per language. (see code: [data-development-code/2-develop-balanced-test-samples.ipynb](data-development-code/2-develop-balanced-test-samples.ipynb))

Together, the dataset that was manually annotated consisted of 1,199 instances.

After annotation, 70 texts (5.83%) were discarded due to being annotated as:
- "do not know": 49 texts (4%)
- "not news": 2 texts (0.1%)
- "multiple" (multiple texts inside one instance): 19 texts (1.58%)

In addition to providing single labels for each instance, the annotator was allowed to provide multiple labels (in a separate column) in cases where two topics were equally present and intertwined in the text. 191 instances were annotated with multiple labels - 16% of all instances. All of these instances were annotated with 2 labels (- there was none which would be annotated with 3 or more labels). This annotation layer was added as additional information but was not used for training and evaluation in the experiments. We use the single-label annotation to calculate inter-annotator agreement and model performance.

The final test dataset comprises 1129 instances.

Distribution of instances per language:

| lang   |   count |
|:-------|--------:|
| hr     |     291 |
| el     |     289 |
| sl     |     282 |
| ca     |     267 |

Label distribution compared to the label distribution of the predictions:

![](figures/test-set-label-distribution-pred-vs-true.png)

Distribution of true labels per language:

![](figures/test-set-label-distribution-per-language.png)

#### Inter-Annotator Agreement

Inter-Annotator Agreement was calculated on a sample of 339 instances, balanced by labels (provided by the 1st annotator). To this end, a second annotator assigned labels to the sample, without having access to the labels provided by GPT-4o model or the 1st annotator.

We calculate the nominal Krippindorff's Alpha.

| pair                        |   nominal Krippendorff Alpha |
|:----------------------------|-----------------------------:|
| pred_GPT4o & 2nd_annotation |                     0.752399 |
| IPTC_true & 2nd_annotation  |                     0.727622 |
| pred_GPT4o & IPTC_true      |                     0.692999 |

Results show satisfactory agreement, higher than 0.667.

## Experiments

To run the experiments, run the following commands:

```bash
TODO: Provide scripts for the experiments
```

### Results



## Available models

This project produced the following models:Multilingual IPTC Media Topic Classifier [classla/multilingual-IPTC-news-topic-classifier](https://huggingface.co/classla/multilingual-IPTC-news-topic-classifier), openly available on Hugging Face.

The model is based on [large-size XLM-RoBERTa model](https://huggingface.co/FacebookAI/xlm-roberta-large) and fine-tuned on a news corpus in 4 languages (Croatian, Slovenian, Catalan and Greek), annotated with the top-level [IPTC NewsCodes Media Topic](https://www.iptc.org/std/NewsCodes/treeview/mediatopic/mediatopic-en-GB.html) labels.

Based on a manually-annotated test set (in Croatian, Slovenian, Catalan and Greek), the model achieves macro-F1 score of 0.746, micro-F1 score of 0.734, and accuracy of 0.734, and outperforms the GPT-4o model (version gpt-4o-2024-05-13) used in a zero-shot setting. If we use only labels that are predicted with a confidence score equal or higher than 0.90, the model achieves micro-F1 and macro-F1 of 0.80.

## Using the trained model

The following script shows how one can use the model:

```python
from transformers import pipeline

# Load a multi-class classification pipeline - if the model runs on CPU, comment out "device"
classifier = pipeline("text-classification", model="classla/multilingual-IPTC-news-topic-classifier", device=0, max_length=512, truncation=True)

# Example texts to classify
texts = [
    """Slovenian handball team makes it to Paris Olympics semifinal Lille, 8 August - Slovenia defeated Norway 33:28 in the Olympic men's handball tournament in Lille late on Wednesday to advance to the semifinal where they will face Denmark on Friday evening. This is the best result the team has so far achieved at the Olympic Games and one of the best performances in the history of Slovenia's team sports squads.""",
    """Moment dog sparks house fire after chewing power bank An indoor monitoring camera shows the moment a dog unintentionally caused a house fire after chewing on a portable lithium-ion battery power bank. In the video released by Tulsa Fire Department in Oklahoma, two dogs and a cat can be seen in the living room before a spark started the fire that spread within minutes. Tulsa Fire Department public information officer Andy Little said the pets escaped through a dog door, and according to local media the family was also evacuated safely. "Had there not been a dog door, they very well could have passed away," he told CBS affiliate KOTV."""]

# Classify the texts
results = classifier(texts)

# Output the results
for result in results:
    print(result)

## Output
## {'label': 'sport', 'score': 0.9985264539718628}
## {'label': 'disaster, accident and emergency incident', 'score': 0.9957459568977356}

```

## Papers

In case you use any of the components for your research, please refer to (and cite) the papers:

```
@book{Kuzman_Ljubesic_2024, place={[Ljubljana}, series={IJS delovno poročilo}, title={Embeddings-based techniques for Media Monitoring Applications (EMMA). WP1, Keyword extraction and topic categorization. Deliverable D1.2, Technical Report on Cross-lingual IPTC News Topic Classification}, note={Nasl. z nasl. zaslona}, publisher={Jožef Stefan Institute]}, author={Kuzman, Taja and Ljubešić, Nikola}, year={2024}, collection={IJS delovno poročilo} }
```

## Work In Progress

- [ ] Journal paper on the topic to be published in IEEE Access journal

## Acknowledgments

**Fundings**: This work was supported by the Slovenian Research and Innovation Agency research project [Embeddings-based techniques for Media Monitoring Applications](https://emma.ijs.si/en/project-plans/) (L2-50070, co-funded by the Kliping d.o.o. agency).

The repository template is based on the template by [cookiecutter]: https://drivendata.github.io/cookiecutter-data-science/
