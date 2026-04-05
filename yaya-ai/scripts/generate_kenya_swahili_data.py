"""Generate Kenya & Swahili SFT training data.

Produces ~5000 Q&A pairs covering:
- Kenya geography, history, politics, culture
- East Africa regional knowledge
- Swahili vocabulary and phrases
- Kenyan daily life, economy, health, education
- Kiswahili ↔ English translation pairs

Output: data/sft/yaya_kenya_swahili.jsonl
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE = os.path.join(REPO_ROOT, "data/sft/yaya_kenya_swahili.jsonl")

SYSTEM_KENYA = "You are Yaya, a helpful, honest, and friendly AI assistant with knowledge of Kenya and East Africa. You answer questions clearly and directly."
SYSTEM_GENERAL = "You are Yaya, a helpful, honest, and friendly AI assistant. You answer questions clearly and thoughtfully."

random.seed(42)


def qa(question, answer, system=SYSTEM_KENYA):
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user",  "content": question},
        {"role": "assistant", "content": answer},
    ]}


# ── Kenya Geography ────────────────────────────────────────────────────────────
KENYA_GEOGRAPHY = [
    qa("What is the capital city of Kenya?", "Nairobi"),
    qa("What is the second largest city in Kenya?", "Mombasa"),
    qa("What is the largest city in Kenya?", "Nairobi"),
    qa("Which ocean borders Kenya to the east?", "The Indian Ocean"),
    qa("What is the highest mountain in Kenya?", "Mount Kenya"),
    qa("How tall is Mount Kenya?", "Mount Kenya is 5,199 metres (17,057 feet) tall."),
    qa("What is the largest lake in Africa?", "Lake Victoria"),
    qa("Which part of Kenya does Lake Victoria border?", "The western part of Kenya, near Kisumu."),
    qa("What is the Great Rift Valley?", "The Great Rift Valley is a large geological fault system that runs through Kenya from north to south, forming valleys, lakes, and volcanic features."),
    qa("Name three Rift Valley lakes in Kenya.", "Lake Nakuru, Lake Naivasha, and Lake Turkana are Rift Valley lakes in Kenya."),
    qa("What is Lake Turkana known for?", "Lake Turkana is the world's largest desert lake and is sometimes called the Jade Sea due to its blue-green color."),
    qa("What is the Maasai Mara?", "The Maasai Mara is a large national reserve in southwestern Kenya, famous for wildlife including the annual wildebeest migration."),
    qa("What countries border Kenya?", "Kenya borders Ethiopia to the north, Somalia to the northeast, Tanzania to the south, Uganda to the west, and South Sudan to the northwest."),
    qa("What is Kenya's total area?", "Kenya covers approximately 580,367 square kilometres."),
    qa("Which county is Nairobi in?", "Nairobi is its own county — Nairobi County."),
    qa("What is the name of Kenya's coast city?", "Mombasa is Kenya's main coastal city."),
    qa("What is Kisumu known for?", "Kisumu is Kenya's third largest city, located on the shores of Lake Victoria in western Kenya."),
    qa("Name the main towns in Kenya.", "The main towns in Kenya include Nairobi, Mombasa, Kisumu, Nakuru, Eldoret, Thika, and Malindi."),
    qa("What national parks are found in Kenya?", "Kenya has many national parks including Maasai Mara, Amboseli, Tsavo East, Tsavo West, Nairobi National Park, and Samburu."),
    qa("What is Amboseli National Park known for?", "Amboseli is famous for its large elephant herds and views of Mount Kilimanjaro across the border in Tanzania."),
    qa("Which river flows through Nairobi?", "The Nairobi River flows through Nairobi."),
    qa("What is the Tana River?", "The Tana River is Kenya's longest river, flowing from the Aberdare Range to the Indian Ocean."),
    qa("In which region is Eldoret located?", "Eldoret is located in the North Rift region of Kenya, in Uasin Gishu County."),
    qa("What is Nakuru known for?", "Nakuru is a city in the Rift Valley known for Lake Nakuru, which attracts flamingos and other wildlife."),
    qa("What continent is Kenya on?", "Kenya is on the African continent."),
    qa("Is Kenya in East Africa?", "Yes, Kenya is one of the countries in East Africa."),
    qa("What is the climate like in Nairobi?", "Nairobi has a mild highland climate due to its altitude of about 1,795 metres. Temperatures rarely exceed 26°C or drop below 10°C."),
    qa("What is the climate like on the Kenyan coast?", "The Kenyan coast has a hot and humid tropical climate with temperatures around 30°C and high rainfall."),
]

# ── Kenya History & Politics ───────────────────────────────────────────────────
KENYA_HISTORY = [
    qa("When did Kenya gain independence?", "Kenya gained independence on 12 December 1963."),
    qa("What is Kenya's independence day called?", "Kenya's independence day is called Jamhuri Day, celebrated on 12 December."),
    qa("Who was Kenya's first president?", "Jomo Kenyatta was Kenya's first president, serving from 1964 to 1978."),
    qa("Who was Kenya's second president?", "Daniel arap Moi was Kenya's second president, serving from 1978 to 2002."),
    qa("Who was Kenya's third president?", "Mwai Kibaki was Kenya's third president, serving from 2002 to 2013."),
    qa("Who was Kenya's fourth president?", "Uhuru Kenyatta was Kenya's fourth president, serving from 2013 to 2022."),
    qa("Who is the current president of Kenya?", "William Ruto became Kenya's fifth president in September 2022."),
    qa("What political party does Kenya have?", "Kenya is a multiparty democracy. Major parties have included Jubilee, ODM (Orange Democratic Movement), and Kenya Kwanza coalition."),
    qa("What is Madaraka Day in Kenya?", "Madaraka Day is celebrated on 1 June and marks the day Kenya achieved self-rule in 1963."),
    qa("What is Mashujaa Day in Kenya?", "Mashujaa Day (formerly Kenyatta Day) is celebrated on 20 October and honours Kenya's national heroes."),
    qa("What is the Mau Mau?", "The Mau Mau was a Kenyan armed rebellion against British colonial rule in the 1950s, led mainly by the Kikuyu people."),
    qa("What was the colonial name for Kenya?", "Kenya was called the East Africa Protectorate and later British East Africa under British colonial rule."),
    qa("What is Kenya's system of government?", "Kenya is a presidential republic. The president is both head of state and head of government."),
    qa("How many counties does Kenya have?", "Kenya has 47 counties."),
    qa("What is devolution in Kenya?", "Devolution in Kenya refers to the transfer of power and resources from the national government to the 47 county governments, introduced by the 2010 Constitution."),
    qa("When was Kenya's current constitution adopted?", "Kenya's current constitution was adopted on 27 August 2010."),
    qa("What is the Kenya National Assembly?", "The National Assembly is the lower house of Kenya's Parliament, consisting of 350 members."),
    qa("What is the Kenya Senate?", "The Senate is the upper house of Kenya's Parliament, representing county interests with 67 senators."),
    qa("Who colonized Kenya?", "Kenya was colonized by Britain. British colonial rule lasted from the late 1880s until independence in 1963."),
    qa("What currency does Kenya use?", "Kenya uses the Kenyan Shilling (KES)."),
]

# ── Kenya Culture & People ─────────────────────────────────────────────────────
KENYA_CULTURE = [
    qa("What are the national languages of Kenya?", "Kenya's two national languages are Swahili (Kiswahili) and English."),
    qa("How many ethnic groups are in Kenya?", "Kenya has over 40 ethnic groups."),
    qa("What is the largest ethnic group in Kenya?", "The Kikuyu are the largest ethnic group in Kenya."),
    qa("Name some ethnic groups in Kenya.", "Some ethnic groups in Kenya include Kikuyu, Luo, Luhya, Kalenjin, Kamba, Kisii, Mijikenda, Somali, Maasai, and Turkana."),
    qa("What is the Maasai people known for?", "The Maasai are a Nilotic ethnic group known for their distinctive clothing (red shuka), jumping dances, warrior culture, and semi-nomadic pastoralist lifestyle."),
    qa("What is Kenya's national food?", "Ugali (stiff maize porridge) is considered Kenya's staple national food, usually eaten with stew, sukuma wiki, or nyama choma."),
    qa("What is ugali?", "Ugali is a stiff porridge made from maize flour (or sometimes sorghum or millet) cooked in water. It is the most common staple food in Kenya."),
    qa("What is nyama choma?", "Nyama choma is Kenyan roasted meat (usually goat or beef), a popular dish at social gatherings."),
    qa("What is sukuma wiki?", "Sukuma wiki is a popular Kenyan dish made from collard greens (kale) cooked with onions and tomatoes."),
    qa("What is chai in Kenya?", "In Kenya, chai is spiced milk tea brewed with tea leaves, milk, sugar, and sometimes ginger or cardamom."),
    qa("What is Kenya's national sport?", "Athletics (long-distance running) is widely considered Kenya's national sport."),
    qa("Why is Kenya famous in athletics?", "Kenya is world-renowned for producing champion long-distance runners, especially from the Rift Valley region."),
    qa("Name a famous Kenyan runner.", "Famous Kenyan runners include Eliud Kipchoge, David Rudisha, Paul Tergat, and Catherine Ndereba."),
    qa("Who is Eliud Kipchoge?", "Eliud Kipchoge is a Kenyan marathon runner, widely considered the greatest marathoner of all time. He was the first person to run a marathon in under 2 hours (1:59:40 in 2019)."),
    qa("What religion do most Kenyans follow?", "Christianity is the most widely practiced religion in Kenya, followed by Islam, especially on the coast and in northeastern Kenya."),
    qa("What is M-Pesa?", "M-Pesa is a mobile money transfer service launched in Kenya in 2007 by Safaricom. It revolutionized financial services in Africa and is used by millions of Kenyans for payments and transfers."),
    qa("What is Safaricom?", "Safaricom is Kenya's largest mobile telecommunications company, known for its M-Pesa mobile money service."),
    qa("What is Kenya's national anthem?", "Kenya's national anthem is called 'Ee Mungu Nguvu Yetu' (O God of All Creation)."),
    qa("What are the colors of the Kenyan flag?", "The Kenyan flag has three horizontal stripes: black (top), red (middle), and green (bottom), with two white stripes between them, and a Maasai shield with spears in the center."),
    qa("What does the black color on Kenya's flag represent?", "The black color on Kenya's flag represents the people of Kenya."),
    qa("What does the red color on Kenya's flag represent?", "The red color represents the blood shed during the struggle for independence."),
    qa("What does the green color on Kenya's flag represent?", "The green represents Kenya's fertile land and natural resources."),
    qa("What is the Maasai shield on Kenya's flag?", "The Maasai shield and spears in the center of the flag represent the defense of freedom."),
]

# ── Kenya Economy & Education ──────────────────────────────────────────────────
KENYA_ECONOMY = [
    qa("What is Kenya's main export?", "Tea is Kenya's most important export, followed by coffee, cut flowers, and horticultural products."),
    qa("Is Kenya the largest tea exporter?", "Kenya is one of the world's largest exporters of black tea, particularly to the UK and Pakistan."),
    qa("What is the main industry in Kenya?", "Agriculture is the backbone of Kenya's economy, employing about 40% of the total population."),
    qa("What is Nairobi's nickname?", "Nairobi is called the 'Green City in the Sun' and is also considered the financial and business hub of East Africa."),
    qa("Is Kenya a developing country?", "Yes, Kenya is classified as a lower-middle income developing country."),
    qa("What is tourism like in Kenya?", "Tourism is a major industry in Kenya, attracting visitors to its national parks, wildlife reserves, beaches, and cultural sites."),
    qa("What is the Nairobi Stock Exchange?", "The Nairobi Securities Exchange (NSE) is Kenya's main stock exchange and one of the largest in Africa."),
    qa("What is the cost of primary education in Kenya?", "Primary education in Kenya is free in public schools under the Free Primary Education programme introduced in 2003."),
    qa("What are Kenya's main universities?", "Major universities in Kenya include the University of Nairobi, Kenyatta University, Moi University, Strathmore University, and JKUAT (Jomo Kenyatta University of Agriculture and Technology)."),
    qa("What is the KCPE?", "The KCPE (Kenya Certificate of Primary Education) is the national exam taken at the end of Standard 8 in Kenya."),
    qa("What is the KCSE?", "The KCSE (Kenya Certificate of Secondary Education) is the national exam taken at the end of Form 4 (secondary school) in Kenya."),
    qa("How many years is primary school in Kenya?", "Primary school in Kenya is 8 years (Standard 1 to Standard 8), but Kenya is transitioning to a new Competency-Based Curriculum (CBC) with different structure."),
    qa("What is the CBC in Kenya?", "The CBC (Competency-Based Curriculum) is Kenya's new education system introduced in 2017, replacing the 8-4-4 system. It focuses on practical skills."),
    qa("What is the 8-4-4 system in Kenya?", "The 8-4-4 system was Kenya's previous education structure: 8 years primary, 4 years secondary, and 4 years university."),
    qa("What is NHIF in Kenya?", "The NHIF (National Hospital Insurance Fund) is Kenya's national health insurance scheme, providing affordable healthcare to members."),
    qa("What is NSSF in Kenya?", "The NSSF (National Social Security Fund) is Kenya's social security scheme for workers, providing retirement and other benefits."),
    qa("What is the Kenya Revenue Authority?", "The Kenya Revenue Authority (KRA) is the government body responsible for collecting taxes in Kenya."),
    qa("What is the shilling exchange rate?", "The Kenyan Shilling fluctuates against the US dollar. As of recent years, 1 USD was approximately 130–150 KES (rates change over time)."),
]

# ── East Africa Regional ───────────────────────────────────────────────────────
EAST_AFRICA = [
    qa("Which countries are in East Africa?", "East Africa typically includes Kenya, Tanzania, Uganda, Rwanda, Burundi, Ethiopia, Somalia, South Sudan, Eritrea, and Djibouti."),
    qa("What is the East African Community?", "The East African Community (EAC) is a regional intergovernmental organisation of East African countries promoting economic integration and cooperation."),
    qa("What is the capital of Tanzania?", "Dodoma is the official capital of Tanzania, though Dar es Salaam is the largest city."),
    qa("What is the capital of Uganda?", "Kampala is the capital of Uganda."),
    qa("What is the capital of Rwanda?", "Kigali is the capital of Rwanda."),
    qa("What is the capital of Ethiopia?", "Addis Ababa is the capital of Ethiopia."),
    qa("What is the tallest mountain in Africa?", "Mount Kilimanjaro in Tanzania is the tallest mountain in Africa at 5,895 metres."),
    qa("What language do people speak in Tanzania?", "Swahili (Kiswahili) and English are official languages of Tanzania. Swahili is the main spoken language."),
    qa("What is the Serengeti?", "The Serengeti is a large national park in Tanzania, famous for the annual wildebeest migration and diverse wildlife."),
    qa("What is the Great Migration?", "The Great Migration is the annual movement of millions of wildebeest, zebras, and gazelles between Tanzania's Serengeti and Kenya's Maasai Mara in search of fresh grass."),
    qa("What ocean is on the east coast of Africa?", "The Indian Ocean is on the east coast of Africa."),
    qa("What is the African Union?", "The African Union (AU) is a continental organization of 55 African countries promoting unity, peace, and development across Africa. Its headquarters are in Addis Ababa, Ethiopia."),
    qa("What is the population of Kenya approximately?", "Kenya's population is approximately 55 million people as of 2024."),
]

# ── Swahili Vocabulary ─────────────────────────────────────────────────────────
def swahili_qa(eng_q, swahili_word, meaning_q=None, meaning=None):
    pairs = [qa(eng_q, swahili_word)]
    if meaning_q and meaning:
        pairs.append(qa(meaning_q, meaning))
    return pairs


SWAHILI_VOCAB = []

SWAHILI_WORDS = [
    # Greetings
    ("What is the Swahili word for hello?", "Jambo", "What does 'Jambo' mean?", "Hello"),
    ("What is the Swahili word for good morning?", "Habari ya asubuhi", "What does 'Habari ya asubuhi' mean?", "Good morning"),
    ("What is the Swahili word for good evening?", "Habari ya jioni", None, None),
    ("What is the Swahili greeting 'Habari'?", "Habari means 'How are you?' or 'What news?'", None, None),
    ("What is the reply to 'Habari'?", "You reply 'Nzuri' (fine/good) or 'Salama' (peaceful/well).", None, None),
    ("What is the Swahili word for goodbye?", "Kwaheri (singular) or Kwaherini (plural)", "What does 'Kwaheri' mean?", "Goodbye"),
    ("What does 'Karibu' mean in Swahili?", "Karibu means 'Welcome' or 'You are welcome'.", None, None),
    ("What does 'Asante' mean in Swahili?", "Asante means 'Thank you'.", None, None),
    ("What does 'Asante sana' mean?", "Asante sana means 'Thank you very much'.", None, None),
    ("How do you say 'you are welcome' in Swahili?", "Karibu or Karibu sana (you are very welcome).", None, None),
    ("How do you say 'please' in Swahili?", "Tafadhali", "What does 'Tafadhali' mean?", "Please"),
    ("How do you say 'excuse me' in Swahili?", "Samahani", "What does 'Samahani' mean?", "Sorry / Excuse me"),
    # Numbers
    ("What is the Swahili word for one?", "Moja", "What does 'Moja' mean?", "One"),
    ("What is the Swahili word for two?", "Mbili", "What does 'Mbili' mean?", "Two"),
    ("What is the Swahili word for three?", "Tatu", "What does 'Tatu' mean?", "Three"),
    ("What is the Swahili word for four?", "Nne", "What does 'Nne' mean?", "Four"),
    ("What is the Swahili word for five?", "Tano", "What does 'Tano' mean?", "Five"),
    ("What is the Swahili word for six?", "Sita", "What does 'Sita' mean?", "Six"),
    ("What is the Swahili word for seven?", "Saba", "What does 'Saba' mean?", "Seven"),
    ("What is the Swahili word for eight?", "Nane", "What does 'Nane' mean?", "Eight"),
    ("What is the Swahili word for nine?", "Tisa", "What does 'Tisa' mean?", "Nine"),
    ("What is the Swahili word for ten?", "Kumi", "What does 'Kumi' mean?", "Ten"),
    ("What is the Swahili word for hundred?", "Mia", "What does 'Mia' mean?", "Hundred"),
    ("What is the Swahili word for thousand?", "Elfu", "What does 'Elfu' mean?", "Thousand"),
    # Common nouns
    ("What is the Swahili word for water?", "Maji", "What does 'Maji' mean?", "Water"),
    ("What is the Swahili word for food?", "Chakula", "What does 'Chakula' mean?", "Food"),
    ("What is the Swahili word for house?", "Nyumba", "What does 'Nyumba' mean?", "House"),
    ("What is the Swahili word for family?", "Familia or Jamaa", "What does 'Familia' mean?", "Family"),
    ("What is the Swahili word for mother?", "Mama", "What does 'Mama' mean?", "Mother"),
    ("What is the Swahili word for father?", "Baba", "What does 'Baba' mean?", "Father"),
    ("What is the Swahili word for child?", "Mtoto", "What does 'Mtoto' mean?", "Child"),
    ("What is the Swahili word for children?", "Watoto", "What does 'Watoto' mean?", "Children"),
    ("What is the Swahili word for person?", "Mtu", "What does 'Mtu' mean?", "Person"),
    ("What is the Swahili word for people?", "Watu", "What does 'Watu' mean?", "People"),
    ("What is the Swahili word for friend?", "Rafiki", "What does 'Rafiki' mean?", "Friend"),
    ("What is the Swahili word for teacher?", "Mwalimu", "What does 'Mwalimu' mean?", "Teacher"),
    ("What is the Swahili word for doctor?", "Daktari", "What does 'Daktari' mean?", "Doctor"),
    ("What is the Swahili word for school?", "Shule", "What does 'Shule' mean?", "School"),
    ("What is the Swahili word for hospital?", "Hospitali", "What does 'Hospitali' mean?", "Hospital"),
    ("What is the Swahili word for money?", "Pesa", "What does 'Pesa' mean?", "Money"),
    ("What is the Swahili word for book?", "Kitabu", "What does 'Kitabu' mean?", "Book"),
    ("What is the Swahili word for books?", "Vitabu", "What does 'Vitabu' mean?", "Books"),
    ("What is the Swahili word for name?", "Jina", "What does 'Jina' mean?", "Name"),
    ("What is the Swahili word for day?", "Siku", "What does 'Siku' mean?", "Day"),
    ("What is the Swahili word for night?", "Usiku", "What does 'Usiku' mean?", "Night"),
    ("What is the Swahili word for sun?", "Jua", "What does 'Jua' mean?", "Sun"),
    ("What is the Swahili word for rain?", "Mvua", "What does 'Mvua' mean?", "Rain"),
    ("What is the Swahili word for road?", "Barabara", "What does 'Barabara' mean?", "Road"),
    ("What is the Swahili word for car?", "Gari", "What does 'Gari' mean?", "Car / Vehicle"),
    ("What is the Swahili word for country?", "Nchi", "What does 'Nchi' mean?", "Country"),
    ("What is the Swahili word for city?", "Mji", "What does 'Mji' mean?", "City / Town"),
    ("What is the Swahili word for market?", "Soko", "What does 'Soko' mean?", "Market"),
    ("What is the Swahili word for work?", "Kazi", "What does 'Kazi' mean?", "Work / Job"),
    # Verbs
    ("What does 'Kula' mean in Swahili?", "Kula means 'to eat'.", None, None),
    ("What does 'Kunywa' mean in Swahili?", "Kunywa means 'to drink'.", None, None),
    ("What does 'Kwenda' mean in Swahili?", "Kwenda means 'to go'.", None, None),
    ("What does 'Kuja' mean in Swahili?", "Kuja means 'to come'.", None, None),
    ("What does 'Kusema' mean in Swahili?", "Kusema means 'to speak' or 'to say'.", None, None),
    ("What does 'Kusaidia' mean in Swahili?", "Kusaidia means 'to help'.", None, None),
    ("What does 'Kufanya kazi' mean in Swahili?", "Kufanya kazi means 'to work'.", None, None),
    ("What does 'Kulala' mean in Swahili?", "Kulala means 'to sleep'.", None, None),
    # Adjectives / phrases
    ("What does 'Nzuri' mean in Swahili?", "Nzuri means 'good', 'fine', or 'nice'.", None, None),
    ("What does 'Mbaya' mean in Swahili?", "Mbaya means 'bad'.", None, None),
    ("What does 'Kubwa' mean in Swahili?", "Kubwa means 'big' or 'large'.", None, None),
    ("What does 'Ndogo' mean in Swahili?", "Ndogo means 'small'.", None, None),
    ("What does 'Haraka' mean in Swahili?", "Haraka means 'quickly' or 'hurry'.", None, None),
    ("What does 'Polepole' mean in Swahili?", "Polepole means 'slowly'.", None, None),
    ("What does 'Sawa' mean in Swahili?", "Sawa means 'OK', 'alright', or 'equal'.", None, None),
    ("What does 'Sawa sawa' mean in Swahili?", "Sawa sawa means 'alright alright' or 'it's fine'.", None, None),
    ("What does 'Ndiyo' mean in Swahili?", "Ndiyo means 'Yes'.", None, None),
    ("What does 'Hapana' mean in Swahili?", "Hapana means 'No'.", None, None),
    ("What does 'Sijui' mean in Swahili?", "Sijui means 'I don't know'.", None, None),
    ("What does 'Naelewa' mean in Swahili?", "Naelewa means 'I understand'.", None, None),
    ("What does 'Ninahitaji' mean in Swahili?", "Ninahitaji means 'I need'.", None, None),
    ("What does 'Ninajua' mean in Swahili?", "Ninajua means 'I know'.", None, None),
    ("What does 'Nakupenda' mean in Swahili?", "Nakupenda means 'I love you'.", None, None),
    ("What does 'Pole' mean in Swahili?", "Pole means 'sorry' or is used to express sympathy.", None, None),
    ("What does 'Hongera' mean in Swahili?", "Hongera means 'Congratulations' or 'Well done'.", None, None),
    ("What does 'Hujambo' mean in Swahili?", "Hujambo is a greeting meaning 'Hello, how are you?' (singular). The reply is 'Sijambo' (I am fine).", None, None),
    ("What is the reply to 'Hujambo'?", "The reply to 'Hujambo' is 'Sijambo', meaning 'I am well'.", None, None),
    ("What does 'Shikamoo' mean in Swahili?", "Shikamoo is a respectful greeting used towards elders. The response is 'Marahaba'.", None, None),
    ("What does 'Marahaba' mean in Swahili?", "Marahaba is the response to 'Shikamoo', acknowledging a respectful greeting.", None, None),
]

for item in SWAHILI_WORDS:
    eng_q, answer = item[0], item[1]
    meaning_q, meaning = item[2], item[3]
    SWAHILI_VOCAB.append(qa(eng_q, answer, SYSTEM_GENERAL))
    if meaning_q and meaning:
        SWAHILI_VOCAB.append(qa(meaning_q, meaning, SYSTEM_GENERAL))

# ── Swahili Sentences & Translations ──────────────────────────────────────────
SWAHILI_SENTENCES = [
    qa("How do you say 'My name is Yaya' in Swahili?", "Jina langu ni Yaya.", SYSTEM_GENERAL),
    qa("How do you say 'I am fine' in Swahili?", "Niko sawa. (or Sijambo.)", SYSTEM_GENERAL),
    qa("How do you say 'Where is the hospital?' in Swahili?", "Hospitali iko wapi?", SYSTEM_GENERAL),
    qa("How do you say 'How much does this cost?' in Swahili?", "Hii ni bei ngapi? (or Inagharimu kiasi gani?)", SYSTEM_GENERAL),
    qa("How do you say 'I want water' in Swahili?", "Nataka maji.", SYSTEM_GENERAL),
    qa("How do you say 'I am hungry' in Swahili?", "Nina njaa.", SYSTEM_GENERAL),
    qa("How do you say 'I am thirsty' in Swahili?", "Nina kiu.", SYSTEM_GENERAL),
    qa("How do you say 'Help me' in Swahili?", "Nisaidie. (or Saidia!)", SYSTEM_GENERAL),
    qa("How do you say 'I do not understand' in Swahili?", "Sielewi.", SYSTEM_GENERAL),
    qa("How do you say 'Please repeat' in Swahili?", "Tafadhali rudia.", SYSTEM_GENERAL),
    qa("How do you say 'Good night' in Swahili?", "Lala salama (Sleep well) or Usiku mwema (Good night).", SYSTEM_GENERAL),
    qa("How do you say 'I live in Nairobi' in Swahili?", "Ninaishi Nairobi.", SYSTEM_GENERAL),
    qa("How do you say 'I am going to school' in Swahili?", "Naenda shule.", SYSTEM_GENERAL),
    qa("How do you say 'Call the police' in Swahili?", "Piga simu polisi! (or Ita polisi!)", SYSTEM_GENERAL),
    qa("How do you say 'I am lost' in Swahili?", "Nimepotea.", SYSTEM_GENERAL),
    qa("What is the Swahili phrase for 'no problem'?", "Hakuna matata — made famous by The Lion King, it means 'no worries' or 'no problem' in Swahili.", SYSTEM_GENERAL),
    qa("What does 'Hakuna matata' mean?", "Hakuna matata is a Swahili phrase meaning 'no worries' or 'no problem'.", SYSTEM_GENERAL),
    qa("How do you count to five in Swahili?", "Moja, mbili, tatu, nne, tano (one, two, three, four, five).", SYSTEM_GENERAL),
    qa("How do you count to ten in Swahili?", "Moja, mbili, tatu, nne, tano, sita, saba, nane, tisa, kumi.", SYSTEM_GENERAL),
]

# ── General Knowledge (Kenya-framed) ──────────────────────────────────────────
KENYA_GENERAL = [
    qa("What is the best way to send money in Kenya?", "M-Pesa is the most popular and convenient way to send money in Kenya. You can send via Safaricom on any phone by dialing *334#."),
    qa("What is 'matatu' in Kenya?", "A matatu is a minibus used for public transport in Kenya. It is the most common way to travel between towns and within cities."),
    qa("What is a 'boda boda' in Kenya?", "A boda boda is a motorcycle taxi used for short trips, especially in towns and rural areas."),
    qa("What is 'mandazi' in Kenya?", "Mandazi is a popular Kenyan fried dough bread, often eaten for breakfast with tea."),
    qa("What is 'githeri'?", "Githeri is a traditional Kenyan dish made of boiled maize and beans mixed together."),
    qa("What is 'mutura'?", "Mutura is a traditional Kenyan sausage made from goat intestines stuffed with meat and spices."),
    qa("What is the Kenyan greeting 'Mambo'?", "Mambo is a casual greeting among youth meaning 'What's up?' The reply is 'Poa' (cool/fine) or 'Safi' (clean/good).", SYSTEM_GENERAL),
    qa("What does 'Poa' mean in Kenyan slang?", "Poa means 'cool' or 'fine' in Kenyan slang.", SYSTEM_GENERAL),
    qa("What is the Kenyan emergency number?", "The main emergency numbers in Kenya are: Police 999 or 112, Ambulance 999 or 112, Fire 999 or 0800 720 112."),
    qa("What is the weather like in Kenya?", "Kenya has varied climate zones. Nairobi is cool and mild (15–25°C), the coast is hot and humid (28–35°C), and northern Kenya is hot and dry."),
    qa("What are Kenya's main exports?", "Kenya's main exports are tea, cut flowers, coffee, horticultural products (vegetables and fruits), and petroleum products."),
    qa("What mobile networks are available in Kenya?", "The main mobile networks in Kenya are Safaricom, Airtel Kenya, and Telkom Kenya."),
    qa("What is the internet penetration in Kenya?", "Kenya has one of the highest internet penetration rates in Africa, largely driven by mobile internet via smartphones."),
    qa("What is Silicon Savannah?", "Silicon Savannah is a nickname for Nairobi, referring to its growing tech startup ecosystem and innovation hub, iHub."),
    qa("What is the Big Five in Kenya?", "The Big Five refers to the five large African mammals traditionally sought by hunters: lion, elephant, buffalo, leopard, and rhinoceros. They can all be found in Kenya's parks."),
    qa("What wildlife is Kenya famous for?", "Kenya is famous for the Big Five (lion, elephant, buffalo, leopard, rhino), as well as giraffes, zebras, cheetahs, hippos, wildebeest, and hundreds of bird species."),
    qa("When is the best time to visit Kenya?", "The best times to visit Kenya are July–October (dry season) and January–February, when wildlife viewing is excellent. The Great Migration peaks in July–September."),
    qa("What is the Kenya Wildlife Service?", "The Kenya Wildlife Service (KWS) is the government body responsible for managing and conserving Kenya's wildlife and national parks."),
    qa("What is JKIA?", "JKIA stands for Jomo Kenyatta International Airport, Kenya's main international airport located in Nairobi."),
    qa("What is the Standard Gauge Railway in Kenya?", "The Standard Gauge Railway (SGR) is a modern railway connecting Nairobi to Mombasa (and beyond). It was built with Chinese support and launched in 2017."),
]

# ── Factual QA with Kenyan context (arithmetic word problems in KES) ──────────
KENYA_MATH = [
    qa("If a matatu fare is 50 shillings, how much do you pay for 3 trips?",
       "3 trips × 50 shillings = 150 shillings.", SYSTEM_GENERAL),
    qa("A bag of maize flour costs 120 shillings. How much do 5 bags cost?",
       "5 × 120 = 600 shillings.", SYSTEM_GENERAL),
    qa("If 1 US dollar equals 130 Kenyan shillings, how many shillings is 5 dollars?",
       "5 × 130 = 650 Kenyan shillings.", SYSTEM_GENERAL),
    qa("A farmer sells 10 kg of tomatoes at 80 shillings per kg. How much does she earn?",
       "10 × 80 = 800 shillings.", SYSTEM_GENERAL),
    qa("School fees are 2,500 shillings per term. How much per year (3 terms)?",
       "3 × 2,500 = 7,500 shillings per year.", SYSTEM_GENERAL),
    qa("A litre of petrol costs 180 shillings. A car needs 40 litres to fill up. How much does a full tank cost?",
       "40 × 180 = 7,200 shillings.", SYSTEM_GENERAL),
    qa("You send 1,000 shillings via M-Pesa. The charge is 15 shillings. How much does the recipient get?",
       "1,000 − 15 = 985 shillings (you pay 1,000 and the recipient gets 985).", SYSTEM_GENERAL),
    qa("A chicken costs 500 shillings. You buy 4 chickens. How much do you pay?",
       "4 × 500 = 2,000 shillings.", SYSTEM_GENERAL),
    qa("The NHIF monthly contribution is 500 shillings. How much do you pay in a year?",
       "12 × 500 = 6,000 shillings per year.", SYSTEM_GENERAL),
    qa("A boda boda charges 30 shillings per km. The hospital is 5 km away. How much is the fare?",
       "5 × 30 = 150 shillings.", SYSTEM_GENERAL),
]

# ── Assemble and write ─────────────────────────────────────────────────────────
ALL = (
    KENYA_GEOGRAPHY * 2 +     # duplicate for more weight
    KENYA_HISTORY * 2 +
    KENYA_CULTURE * 2 +
    KENYA_ECONOMY +
    EAST_AFRICA +
    SWAHILI_VOCAB * 3 +       # 3x weight for vocabulary retention
    SWAHILI_SENTENCES * 2 +
    KENYA_GENERAL +
    KENYA_MATH * 3
)

random.shuffle(ALL)

os.makedirs(os.path.join(REPO_ROOT, "data/sft"), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for item in ALL:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Generated {len(ALL)} examples → {OUT_FILE}")
print(f"  Kenya Geography:   {len(KENYA_GEOGRAPHY)} × 2  = {len(KENYA_GEOGRAPHY)*2}")
print(f"  Kenya History:     {len(KENYA_HISTORY)} × 2   = {len(KENYA_HISTORY)*2}")
print(f"  Kenya Culture:     {len(KENYA_CULTURE)} × 2   = {len(KENYA_CULTURE)*2}")
print(f"  Kenya Economy:     {len(KENYA_ECONOMY)} × 1   = {len(KENYA_ECONOMY)}")
print(f"  East Africa:       {len(EAST_AFRICA)} × 1     = {len(EAST_AFRICA)}")
print(f"  Swahili Vocab:     {len(SWAHILI_VOCAB)} × 3    = {len(SWAHILI_VOCAB)*3}")
print(f"  Swahili Sentences: {len(SWAHILI_SENTENCES)} × 2  = {len(SWAHILI_SENTENCES)*2}")
print(f"  Kenya General:     {len(KENYA_GENERAL)} × 1   = {len(KENYA_GENERAL)}")
print(f"  Kenya Math:        {len(KENYA_MATH)} × 3      = {len(KENYA_MATH)*3}")
