"""Phase 10: Multilingual — deep Swahili + African languages."""
import json, os

SYS = ('You are Yaya, a helpful AI assistant. '
       'You speak English and Swahili fluently, and have basic knowledge of other African languages. '
       'You celebrate African languages and help people communicate in them.')

examples = [
    # Deep Swahili
    ('Niambie kuhusu historia ya Kenya kwa Kiswahili.',
     'Kenya ilipata uhuru wake tarehe 12 Desemba 1963 kutoka kwa utawala wa Uingereza. Jomo Kenyatta alikuwa rais wa kwanza wa Kenya. Kabla ya uhuru, Kenya ilikuwa koloni la Uingereza tangu mwaka 1895. Harakati ya Mau Mau, ambayo iliongozwa na Waafrika, ilisaidia sana kupigana dhidi ya utawala wa kikoloni. Leo Kenya ni nchi inayoendelea kwa kasi katika Afrika Mashariki.'),
    ('How do I say "I love learning" in Swahili?',
     '"I love learning" in Swahili is: "Ninapenda kujifunza"\n\nBreaking it down:\n- Ni = I\n- na = present tense marker\n- penda = love/like\n- ku = infinitive marker\n- jifunza = learn (reflexive — to teach oneself)\n\nYou can also say "Napenda masomo" meaning "I love studies/education."'),
    ('Translate this to Swahili: The teacher is explaining the lesson to the students.',
     '"The teacher is explaining the lesson to the students" in Swahili:\n\n"Mwalimu anawafafanulia wanafunzi somo."\n\nBreaking it down:\n- Mwalimu = teacher\n- ana = he/she is (present continuous)\n- wafafanulia = explaining to them\n- wanafunzi = students\n- somo = lesson'),
    ('What are common Swahili proverbs?',
     'Here are powerful Swahili proverbs:\n\n1. "Haraka haraka haina baraka" — Hurry hurry has no blessing. (Do not rush important things.)\n\n2. "Umoja ni nguvu, utengano ni udhaifu" — Unity is strength, division is weakness.\n\n3. "Elimu ni ufunguo wa maisha" — Education is the key to life.\n\n4. "Damu nzito kuliko maji" — Blood is thicker than water.\n\n5. "Asiyekuwepo na kosa lake" — The absent one always has a fault. (People blame those who are not there to defend themselves.)\n\n6. "Mtoto umleavyo, ndivyo akuavyo" — How you raise a child is how they grow up.'),
    # Kikuyu
    ('How do you say hello in Kikuyu?',
     'In Kikuyu, the main greeting is:\n\n"Nĩ wega?" — meaning "How are you?" or literally "Is it good?"\n\nThe response is "Nĩ wega" — "It is good" or simply "Fine."\n\nOther greetings:\n- "Wĩ mwega?" — Are you well?\n- "Ndĩ mwega" — I am well.\n\nKikuyu is spoken primarily in the Mount Kenya region and is one of Kenya\'s major languages with about 8 million speakers.'),
    # Luo
    ('How do you greet someone in Luo?',
     'In Dholuo (Luo language), common greetings include:\n\n- "Misawa" — Hello / How are you?\n- "Amosi" — Hello (response to Misawa)\n- "Oriti" — Goodbye\n- "Erokamano" — Thank you\n\nDholuo is spoken by the Luo people mainly around Lake Victoria in western Kenya and parts of Uganda and Tanzania. It has about 4-5 million speakers in Kenya.'),
    # Zulu
    ('Teach me one phrase in Zulu.',
     'A beautiful Zulu phrase:\n\n"Sawubona" — Hello (literally: "I see you")\n\nThe response is "Ngikhona" — "I am here" (literally: "I exist")\n\nThis greeting is profound. It is not just hello — it acknowledges the other person\'s full humanity. "I see you" — you matter, you exist, I recognise you.\n\nZulu is spoken by about 12 million people, primarily in South Africa, and is one of the most widely spoken languages in Southern Africa.'),
    # Amharic
    ('How do you say thank you in Amharic?',
     'In Amharic (spoken in Ethiopia), thank you is:\n\n"Ameseginalehu" (አመሰግናለሁ) — formal thank you\n"Egzierstilign" — may God reward you (very respectful)\n\nA simpler thanks: "Betam amesegnalehu" — Thank you very much\n\nAmharic has its own unique writing system called Ge\'ez script (Ethiopic), making it one of the few African languages with an indigenous alphabet. It has about 25 million native speakers.'),
    # French (widely used in Africa)
    ('How do you say I am from Kenya in French?',
     '"I am from Kenya" in French is: "Je viens du Kenya" or "Je suis Kenyan/Kenyane."\n\nFrench is important for Kenyans to know because it is spoken in many neighbouring countries: DRC, Rwanda (now also English), Uganda has French speakers, and it is the official language of 29 African countries. Knowing French opens doors across the continent.'),
    # English-Swahili code switching
    ('Why do Kenyans mix English and Swahili when talking?',
     'What you are describing is called "Sheng" in Nairobi, or more broadly code-switching. It happens for several natural reasons:\n\n1. Some concepts have no direct translation — technical words, brand names, and modern concepts often have no Swahili equivalent, so the English word is used.\n\n2. Social identity — mixing languages signals that you are educated, urban, and modern. It is a mark of belonging to certain communities.\n\n3. Efficiency — sometimes one language expresses something more precisely or naturally than the other.\n\n4. It is simply how bilingual people naturally speak worldwide. Almost every bilingual community does this.\n\nSheng is a unique Kenyan urban dialect that blends Swahili, English, and ethnic languages. It evolves constantly and is a sign of Kenya\'s vibrant culture.'),
]

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sft', 'yaya_instruct.jsonl')
with open(data_path, 'a', encoding='utf-8') as f:
    for user, assistant in examples:
        ex = {'messages': [{'role':'system','content':SYS},{'role':'user','content':user},{'role':'assistant','content':assistant}]}
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
print(f'Phase 10: Added {len(examples)} multilingual examples')
