"""Phase 2: Instruction Following & Multi-Turn Conversation data generator.

Produces ~3500 multi-turn conversations (3-6 turns each) teaching Yaya to:
- Maintain context across turns
- Follow complex multi-step instructions
- Handle follow-up questions naturally
- Use format instructions (lists, numbered, paragraph)
- Ask clarifying questions when needed
- Reference earlier parts of the conversation

Output: data/sft/yaya_phase2_multiturn.jsonl
"""

import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE = os.path.join(REPO_ROOT, "data/sft/yaya_phase2_multiturn.jsonl")

SYS = "You are Yaya, a helpful, honest, and friendly AI assistant. You answer clearly, maintain context across the conversation, and follow instructions precisely."
SYS_KE = "You are Yaya, a helpful AI assistant with knowledge of Kenya and East Africa. You are friendly, clear, and maintain context throughout the conversation."

random.seed(42)

def conv(*turns, system=SYS):
    """Build a conversation from alternating user/assistant turns."""
    messages = [{"role": "system", "content": system}]
    for i, text in enumerate(turns):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
    return {"messages": messages}


# ── Category 1: Science & Nature follow-ups ───────────────────────────────────
SCIENCE_CONVOS = [
    conv(
        "What is photosynthesis?",
        "Photosynthesis is the process plants use to convert sunlight, water, and carbon dioxide into glucose (sugar) and oxygen. It happens in the chloroplasts of plant cells, using a green pigment called chlorophyll.",
        "How does that benefit the plant?",
        "The glucose produced serves as the plant's food — it provides energy for growth, reproduction, and all biological functions. Without photosynthesis, the plant would have no fuel to survive.",
        "And where does the oxygen go?",
        "The oxygen is released as a byproduct into the atmosphere through tiny pores in leaves called stomata. This is actually the primary source of breathable oxygen on Earth, which is why plants are essential for all animal life."
    ),
    conv(
        "Explain the water cycle.",
        "The water cycle is the continuous movement of water through Earth's systems. It has four main stages: (1) Evaporation — water from oceans, lakes, and rivers is heated by the sun and turns into water vapor. (2) Condensation — water vapor rises, cools, and forms clouds. (3) Precipitation — water falls back to Earth as rain or snow. (4) Collection — water collects in oceans, rivers, and groundwater, then the cycle repeats.",
        "What causes evaporation to speed up?",
        "Several factors speed up evaporation: higher temperature (more heat energy), lower humidity (drier air absorbs water faster), wind (carries away water vapor so more can evaporate), and larger surface area exposed to air.",
        "Does this cycle affect Kenya's climate?",
        "Yes, significantly. Kenya's long rains (March–May) and short rains (October–December) are driven by the water cycle interacting with the Indian Ocean. The ocean evaporates massive amounts of water, forming monsoon winds that bring rainfall inland. The Rift Valley lakes also create local evaporation-condensation cycles that affect rainfall in surrounding areas."
    ),
    conv(
        "What is gravity?",
        "Gravity is a fundamental force that attracts any two objects with mass toward each other. The greater the mass and the closer the distance, the stronger the gravitational pull. On Earth, gravity pulls everything toward the center of the planet, giving us weight and keeping the atmosphere in place.",
        "Why does the moon not fall into Earth then?",
        "The Moon is actually falling toward Earth constantly — but it's also moving sideways so fast that it keeps 'missing' the planet. This balance between falling inward and moving sideways creates a stable orbit. The same principle applies to all satellites, including man-made ones.",
        "How fast does the Moon move?",
        "The Moon travels at about 1 kilometre per second (3,600 km/h) in its orbit around Earth. At that speed, it takes approximately 27.3 days to complete one full orbit. Its relatively slow speed compared to artificial satellites is because it orbits much farther away — about 384,000 km from Earth."
    ),
    conv(
        "What causes earthquakes?",
        "Earthquakes are caused by the movement of tectonic plates — the large pieces of rock that make up Earth's outer shell. When these plates grind past each other, collide, or pull apart, stress builds up along fault lines. When that stress releases suddenly, it sends seismic waves through the ground, causing shaking.",
        "Is Kenya at risk of earthquakes?",
        "Yes, Kenya sits along the East African Rift System, one of the world's most active geological zones. The Rift Valley is literally splitting Africa apart — very slowly (a few centimetres per year). This creates regular small tremors, especially around Nakuru, Naivasha, and along the Rift Valley. Major destructive earthquakes are less common but do occur.",
        "What should you do during an earthquake?",
        "The standard advice is: Drop, Cover, and Hold On. Get under a sturdy table or desk, cover your head and neck, and hold on until the shaking stops. Stay away from windows, heavy furniture that could topple, and outdoor power lines. If you are outside, move away from buildings and stay in the open."
    ),
    conv(
        "How does the human immune system work?",
        "Your immune system is your body's defence network. It has two main components: the innate immune system (fast, general defence — skin, fever, inflammation) and the adaptive immune system (targeted, specific — creates antibodies against particular pathogens). When a foreign invader like a bacteria or virus enters, white blood cells detect it, signal an alarm, and launch an attack. The adaptive system also 'remembers' past infections, which is why you rarely get chicken pox twice.",
        "How do vaccines relate to this?",
        "Vaccines train your immune system without causing the actual disease. They introduce a weakened, killed, or partial version of a pathogen (or instructions to make a harmless protein from it). Your immune system responds, creates antibodies, and most importantly, creates memory cells. If you later encounter the real pathogen, your immune system recognises it immediately and destroys it before it can make you seriously ill.",
        "Are vaccines safe?",
        "Yes, vaccines are among the most rigorously tested medical interventions. Before approval, they go through multiple phases of clinical trials involving tens of thousands of people. The benefits — preventing serious illness, disability, and death — vastly outweigh the risks for all approved vaccines. Minor side effects like a sore arm or mild fever are signs the immune system is responding, which is expected and normal."
    ),
    conv(
        "What is DNA?",
        "DNA (Deoxyribonucleic Acid) is the molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all living organisms. It is shaped like a twisted ladder — a double helix — with the rungs made of pairs of four chemical bases: Adenine, Thymine, Guanine, and Cytosine. The sequence of these bases is the 'code' that determines your traits.",
        "Where is DNA found in the body?",
        "Almost every cell in your body contains DNA, stored in the nucleus. The full set of DNA in a cell is called a genome. If you stretched out all the DNA from a single human cell, it would be about 2 metres long — yet it is coiled and packed so tightly it fits in a nucleus just 6 micrometres wide.",
        "What is genetic inheritance?",
        "You inherit DNA from both parents — roughly half from your mother and half from your father. This is why you share traits with both parents but are not identical to either. Some traits follow simple patterns (one gene determines the outcome), while others are complex (many genes interact with the environment). This is why siblings from the same parents can look quite different."
    ),
    conv(
        "What is climate change?",
        "Climate change refers to long-term shifts in global temperatures and weather patterns. While some climate change is natural, since the mid-20th century human activities — especially burning fossil fuels like coal, oil, and gas — have become the primary driver. These activities release greenhouse gases (mainly CO2 and methane) that trap heat in the atmosphere, causing the planet to warm.",
        "How does it affect Kenya specifically?",
        "Kenya is already experiencing significant climate change impacts: more frequent and severe droughts (devastating agriculture and pastoralism), unpredictable rainfall (farmers can no longer rely on seasonal patterns), rising temperatures (increasing water scarcity and affecting crop yields), and more intense flooding. Northern Kenya and the arid regions are most severely affected.",
        "What is Kenya doing about it?",
        "Kenya has made several commitments: The government has pledged to reduce greenhouse gas emissions by 32% by 2030 under the Paris Agreement. Kenya already gets over 90% of its electricity from renewable sources (geothermal, hydro, wind, solar). The Olkaria geothermal plant in the Rift Valley is one of the largest in Africa. Kenya is also planting billions of trees and investing in climate-smart agriculture."
    ),
]

# ── Category 2: Practical / how-to multi-step ─────────────────────────────────
HOWTO_CONVOS = [
    conv(
        "How do I start a small business in Kenya?",
        "Starting a business in Kenya involves these key steps: (1) Choose your business idea and validate it — make sure there is real demand. (2) Register your business with the Business Registration Service at Huduma Centre or online at businessregistration.go.ke. (3) Get a KRA PIN for tax purposes. (4) Open a business bank account. (5) Get any required licences for your specific industry. (6) Set up your M-Pesa business account (Lipa na M-Pesa) for payments.",
        "How much does registration cost?",
        "Business name registration costs around Ksh 950 and takes about 3-5 working days. If you are registering a limited company (Ltd), costs are higher — approximately Ksh 10,000–20,000 depending on complexity. You can do everything online now through ecitizen.go.ke, which has made the process much faster.",
        "What type of business is easiest to register?",
        "A Sole Proprietorship (business name) is the simplest and cheapest option. You just register a business name under your own name. The downside is there is no legal separation between you and the business — if the business has debts, you are personally liable. A Partnership involves 2+ people sharing ownership. A Limited Company gives legal protection and looks more professional, which matters if you want to work with large organisations or get loans."
    ),
    conv(
        "How do I save money on a small income?",
        "Saving on a small income requires a clear system. Start with the 50/30/20 rule adapted for Kenya: spend 50% on essentials (rent, food, transport), 30% on wants, and save at least 20%. Use M-Pesa's M-Shwari or Stanbic's Goal account to lock savings so you are not tempted to spend. Cut costs by buying food in bulk from markets rather than supermarkets, and cook at home rather than eating out.",
        "What about chama savings groups?",
        "Chama (savings group) is one of the most effective savings tools in Kenya. A group of people contribute a fixed amount weekly or monthly. Members take turns receiving the pooled amount (merry-go-round) or the group invests collectively. The social accountability — knowing others expect your contribution — makes you more disciplined than saving alone. Many successful Kenyan businesses started from chama savings.",
        "How do I choose a good chama?",
        "Look for these qualities in a chama: Clear written rules about contributions, meetings, and consequences for missing payments. Registered as a self-help group (SHG) — this gives legal standing. Transparent record-keeping — minutes, accounts, receipts. Trust — start with close friends or family, or join an established group with a good track record. Keep the group small (10-20 people) to maintain accountability."
    ),
    conv(
        "How do I prepare for a job interview?",
        "Prepare in four areas: (1) Research — learn about the company, their products, mission, and recent news. (2) Your story — prepare clear answers for 'Tell me about yourself,' 'Why do you want this job,' and 'What are your strengths and weaknesses.' (3) Examples — use the STAR method (Situation, Task, Action, Result) for experience questions. (4) Questions to ask — always have 2-3 questions ready for the interviewer.",
        "What should I wear?",
        "Dress one level above what the role requires. For an office job, wear formal business attire — for men, a clean suit or shirt and trousers; for women, a suit, formal dress, or blouse with trousers. Clothes should be clean, ironed, and well-fitting. Avoid strong perfume. First impressions form in seconds, and appearing put-together signals you take the opportunity seriously.",
        "What if they ask about salary expectations?",
        "Research the salary range for that role in Kenya before the interview — ask friends in the industry, check LinkedIn jobs, or use Glassdoor. When asked, give a range based on your research, not a specific number: 'Based on my experience and market rates for this role, I am targeting between Ksh 80,000 and 100,000, but I am open to discussing.' Never name a figure too low, as it is hard to negotiate up later."
    ),
    conv(
        "How do I build a vegetable garden at home?",
        "Start small. Choose a spot that gets at least 6 hours of sunlight daily. For beginners, raised beds (or old car tyres, sacks, or boxes) work better than in-ground planting — better drainage and easier to manage. Start with fast-growing, easy vegetables: sukuma wiki (kale), spinach, tomatoes, or onions. Buy good seedlings or seeds from an agrovets shop.",
        "What soil do I need?",
        "A good growing mix is: 1 part red soil (topsoil), 1 part compost (decomposed leaves, kitchen vegetable scraps, or manure), and 1 part sand for drainage. Avoid soil from construction sites — it lacks nutrients. You can buy ready-mixed potting soil from agrovets. Add a handful of DAP fertilizer when planting, then top-dress with CAN fertilizer after 3 weeks.",
        "How often do I water it?",
        "Water in the morning or evening, never at midday when the sun evaporates water before it soaks in. Most vegetables need watering every 1-2 days in dry weather, less in rainy weather. Push your finger 2 cm into the soil — if it feels dry, water. A simple drip system using plastic bottles with small holes (buried next to plant roots) saves water and keeps soil consistently moist."
    ),
    conv(
        "I want to learn programming. Where do I start?",
        "Start with Python — it has the simplest syntax (reads almost like English) and is used in web development, data science, AI, and automation. Begin with the absolute basics: variables (storing information), data types (numbers, text, lists), and control flow (if/else, loops). Free resources: Python.org has an official tutorial; freeCodeCamp on YouTube has full courses; Codecademy has interactive lessons.",
        "How long will it take to get a job?",
        "With consistent daily practice (1-2 hours), you can reach employable skill level in 6-12 months. The path typically is: Month 1-2 — Python basics. Month 3-4 — Build 2-3 small projects (a calculator, a to-do app). Month 5-8 — Learn a web framework (Flask or Django) or data tools (pandas, NumPy). Month 9-12 — Build a portfolio of 3-5 projects and start applying. The key is building real projects, not just following tutorials.",
        "What jobs are available in Kenya for programmers?",
        "Kenya's tech sector is growing rapidly. High-demand roles include: Software Developer (web, mobile — 1 and Android), Data Analyst/Scientist, DevOps Engineer, and Cybersecurity specialist. Companies hiring include Safaricom, KCB, Equity Bank, NCBA, iHub startups, and international remote-work companies. Junior developers in Nairobi earn Ksh 40,000-80,000; senior developers Ksh 150,000-400,000+. Remote work for international companies pays 3-5x more."
    ),
    conv(
        "How can I improve my English writing?",
        "Focus on three things: reading, practice, and feedback. Read widely — news articles, books, online essays — and pay attention to sentence structure. Write every day, even just a journal entry or WhatsApp messages with correct grammar. Use Grammarly (free version) to catch errors and explain why they are mistakes. Start with basic rules: subject-verb agreement, apostrophes, and comma use.",
        "What are common English mistakes Kenyans make?",
        "Common Kenyan English mistakes include: (1) Overusing 'kindly' — it sounds formal and slightly odd in casual contexts; use 'please' instead. (2) 'I am going to school going' — unnecessary repetition from direct Swahili translation. (3) 'My name is called John' — correct form is 'My name is John.' (4) Adding '-s' to uncountable nouns: 'informations', 'furnitures', 'advices' — these have no plural form. (5) 'I have been knowing him' — should be 'I have known him.'",
        "How do I sound more professional in emails?",
        "Professional emails follow this structure: Clear subject line (specific, not 'Hi'). Greeting (Dear [Name], / Good morning [Name],). One-sentence purpose (I am writing regarding...). Clear body — one idea per paragraph. Action item (Please confirm by...). Closing (Kind regards, / Yours sincerely,). Avoid: slang, excessive exclamation marks, and very long sentences. Read your email out loud before sending — if it sounds awkward, rewrite it."
    ),
]

# ── Category 3: Personal / emotional support ──────────────────────────────────
SUPPORT_CONVOS = [
    conv(
        "I am feeling very stressed about my exams. What should I do?",
        "Exam stress is completely normal, and the fact that you care about it shows you take your education seriously. First, break the problem down: list every subject and topic you need to cover, then create a study timetable. Seeing a clear plan makes the big task feel smaller and more manageable.",
        "I have tried timetables before but I never stick to them.",
        "That is a very common experience. The reason most timetables fail is they are too ambitious — they assume every hour is productive and leave no buffer. Try this instead: plan in 45-minute blocks with 15-minute breaks (the Pomodoro technique). Set just 3 study goals per day, not 10. And plan your hardest subject first, when your mind is sharpest. Most importantly, put the phone in another room when studying.",
        "What if I blank out during the exam?",
        "Blanking out is your brain's stress response interfering with memory retrieval. If it happens: take 3 deep slow breaths (this activates the calming nervous system), write down anything you do remember on the answer sheet — even keywords, skip the question and come back later (sometimes retrieval needs a different starting point), and focus on what you know, not what you do not. Preparation is the best defence: the more you practise under timed conditions before the exam, the less novel the situation feels."
    ),
    conv(
        "I had a big argument with my best friend. What should I do?",
        "Arguments with close friends can feel devastating because the relationship means a lot to you. Before doing anything, give yourself and them a little time to cool down — attempting to resolve things while emotions are still high usually makes it worse.",
        "How long should I wait?",
        "There is no exact rule, but a few hours to a day is usually enough for most disagreements. If the argument was very intense, a day or two can help. The goal is not to give the silent treatment as punishment, but to allow both sides to think more clearly. Check in with yourself: what are you actually upset about? Sometimes the surface argument is not the real issue.",
        "How do I bring it up without starting another fight?",
        "Use 'I' statements instead of 'You' statements. Instead of 'You always dismiss my feelings,' say 'I felt hurt when my opinion was dismissed.' Start by acknowledging any part you played, even if you were mostly right — this lowers their defences. Listen to understand, not just to respond. End by affirming the friendship: 'I value our friendship, which is why I want to sort this out.' Most strong friendships survive arguments — they can even become stronger after resolving conflict well."
    ),
    conv(
        "My business is failing and I do not know what to do.",
        "I am sorry to hear that — it is an incredibly stressful situation. The first step is to get clear on what 'failing' means exactly. Is it running out of cash? No customers? Unprofitable even with customers? Debt? Each problem has a different solution, so identifying the specific issue is essential before taking action.",
        "I have customers but not enough to cover costs.",
        "That is actually a more fixable situation than having no customers at all. Three paths to consider: First, increase revenue — can you serve existing customers more (upsell), charge slightly more, or find new customers from your existing network? Second, cut costs — review every expense and cut anything not directly generating revenue. Third, pivot slightly — could a small change in what you offer attract more of the customers you already have? Sometimes a very minor adjustment (a new product, a new location, a different marketing message) changes everything.",
        "I am thinking of giving up entirely.",
        "That is an understandable feeling when you are exhausted and under pressure. But before deciding, consider whether you have truly explored all options or whether you are making this decision from a place of stress and exhaustion. Talk to someone who has run a business — a mentor, a family member who is entrepreneurial, or a member of a business association. Sometimes an outsider sees a clear path you cannot see from inside the problem. If after careful reflection you decide to close, that is also a valid and sometimes brave choice — it frees you to start something better with what you have learned."
    ),
    conv(
        "How can I help my child do better in school?",
        "You are already doing the most important thing — showing you care about your child's education. Research consistently shows that parental involvement is one of the strongest predictors of academic success, regardless of income. Start by creating a regular study time at home — the same time every day builds a habit. Make sure there is a quiet place to study, even if it is just a corner of a room.",
        "My child says school is boring.",
        "That feeling is worth taking seriously. Ask your child which subjects interest them and try to connect school topics to those interests. If they love football, use football statistics to practise maths. If they love cooking, use recipes for fractions and measurement. Visit the library or find educational YouTube channels on topics they find interesting. Boredom often comes from not seeing the relevance of what they are learning.",
        "What about getting a tutor?",
        "A tutor can be very helpful, especially for specific subjects where your child is struggling. Look for tutors through recommendations from other parents, local secondary school students who perform well (often affordable and relatable), or online platforms. Make sure to sit in on the first session to assess the approach. The best tutors explain concepts in multiple ways until the child understands — not just repeat the same explanation louder."
    ),
]

# ── Category 4: Format-following conversations ────────────────────────────────
FORMAT_CONVOS = [
    conv(
        "Give me 5 tips for better sleep.",
        "Here are 5 evidence-based tips for better sleep:\n\n1. Keep a consistent schedule — go to bed and wake up at the same time every day, even weekends.\n2. Make your room dark and cool — your body temperature naturally drops during sleep, so a cooler room (17-20°C) helps.\n3. Avoid screens for 30-60 minutes before bed — the blue light from phones and laptops delays melatonin production.\n4. Avoid caffeine after 2 pm — coffee and tea have a half-life of 5-6 hours and will still be in your system at bedtime.\n5. Wind down with a routine — reading, light stretching, or a warm shower signals to your brain that sleep is coming.",
        "Can you expand on tip 3 about screens?",
        "Of course. Blue light (wavelengths around 480nm) is particularly effective at suppressing melatonin — the hormone that makes you feel sleepy. This is because blue light is the dominant wavelength in daylight, and your brain interprets it as a signal to stay awake. When you use a phone or laptop at night, your brain essentially thinks it is still daytime. The effect is not just delayed sleep — studies show it also reduces sleep quality even after you do fall asleep.\n\nPractical solutions: Use 'Night Mode' or 'Warm Light' settings on your devices in the evening. Download f.lux (free) which automatically shifts your screen to warmer tones after sunset. Or simply put the phone in another room an hour before bed — this also removes the temptation to check notifications.",
        "What if I cannot sleep even without screens?",
        "Try a technique called stimulus control: only use your bed for sleep — do not work, watch TV, or scroll social media in bed. If you have been lying awake for more than 20 minutes, get up and do a calm, boring activity in dim light (reading a physical book works well) until you feel sleepy, then return to bed. This trains your brain to associate the bed with sleep rather than wakefulness. Another option is the 4-7-8 breathing technique: inhale for 4 counts, hold for 7, exhale for 8 — it activates the parasympathetic nervous system and can induce drowsiness within minutes."
    ),
    conv(
        "List the pros and cons of moving to Nairobi for work.",
        "Here are the main pros and cons of moving to Nairobi for work:\n\n**PROS:**\n- Economic opportunity: Nairobi is East Africa's financial hub — more jobs, higher salaries, and more industries than most other Kenyan cities\n- Career growth: headquarters of major banks, tech companies, NGOs, and multinationals are here\n- Infrastructure: international airport, SGR rail, reliable internet, and good roads in major areas\n- Lifestyle: diverse entertainment, restaurants, hospitals, and universities\n- Networking: proximity to decision-makers and industry professionals\n\n**CONS:**\n- High cost of living: rent, food, and transport are significantly more expensive than upcountry\n- Traffic: commuting can consume 2-4 hours per day in bad areas\n- Safety concerns: some areas have higher crime rates, especially at night\n- Pollution: air quality and noise are significant in the CBD and industrial areas\n- Distance from family: social support networks are left behind",
        "What about housing costs specifically?",
        "Housing in Nairobi varies enormously by area. Rough monthly rent ranges:\n\n- **Westlands, Kilimani, Lavington**: Ksh 30,000–80,000+ (studio to 2BR) — modern, central, popular with professionals\n- **Ruaka, Kiambu Road**: Ksh 15,000–35,000 — good value, slightly outside Nairobi but connected\n- **Kasarani, Roysambu**: Ksh 8,000–25,000 — more affordable, busy area near Thika Road\n- **Eastlands (Umoja, Donholm)**: Ksh 6,000–18,000 — most affordable but longer commute to CBD\n- **Ngong, Rongai**: Ksh 7,000–20,000 — quieter suburban areas with longer commutes\n\nAs a rule of thumb, rent should not exceed 30% of your net salary. If your take-home is Ksh 60,000, target rent of Ksh 15,000–18,000."
    ),
    conv(
        "Write a short professional bio for a software developer.",
        "Here is a professional bio suitable for LinkedIn or a portfolio:\n\n---\n\n**[Your Name]** is a software developer with [X] years of experience building web and mobile applications. Based in Nairobi, Kenya, [he/she/they] specialises in Python, JavaScript, and cloud infrastructure, with a focus on scalable solutions for African markets.\n\n[He/She/They] has worked with startups and established companies across fintech, agritech, and e-commerce, consistently delivering products that are fast, reliable, and user-friendly.\n\nWhen not coding, [Name] contributes to open-source projects and mentors emerging developers in the Nairobi tech community.\n\n---\n\nFeel free to fill in your name, years of experience, and pronouns. Would you like me to adjust the tone — more formal, more casual, or focused on a specific industry?",
        "Make it more casual and focused on fintech.",
        "Here's a more casual, fintech-focused version:\n\n---\n\nI'm a software developer passionate about building financial tools that actually work for everyday Kenyans. Over the past [X] years, I've shipped products in mobile money, lending, and banking — always with the goal of making financial services faster, cheaper, and more accessible.\n\nI work primarily with Python and JavaScript, and I genuinely enjoy the challenge of building systems that handle real money at scale — where bugs have consequences.\n\nBased in Nairobi. Always happy to talk fintech, engineering challenges, or the future of digital payments in Africa.\n\n---\n\nThis version sounds more like a real person talking rather than a formal document, which works well on LinkedIn and at tech meetups."
    ),
    conv(
        "Summarise the history of Kenya in 3 paragraphs.",
        "**Pre-colonial Kenya** was home to diverse peoples including Bantu-speaking communities (Kikuyu, Luhya, Kamba), Nilotic groups (Luo, Kalenjin, Maasai), and Cushitic peoples in the north. Arab traders established settlements along the Swahili Coast from the 8th century, creating a rich trading culture. The interior remained largely autonomous until the arrival of European explorers and missionaries in the 19th century.\n\n**Colonial period (1895–1963)**: Britain declared Kenya a protectorate in 1895, later renaming it the East Africa Protectorate (1920: Kenya Colony). The British built the Uganda Railway (Mombasa to Uganda), displacing communities and opening the interior to settlers who seized fertile highland land. Resistance grew through the Mau Mau uprising (1952–1960), an armed liberation movement led largely by the Kikuyu. International pressure and mounting costs of suppression led Britain to begin decolonisation.\n\n**Independent Kenya (1963–present)**: Kenya gained independence on 12 December 1963, with Jomo Kenyatta as first prime minister and later first president. The country maintained a single-party system under KANU for decades. The 1990s brought multiparty democracy; the 2007–08 post-election violence killed over 1,000 people and led to a coalition government. The 2010 constitution introduced a devolved system of 47 counties. Today Kenya is East Africa's largest economy and is known for technology innovation (M-Pesa, Silicon Savannah), long-distance running excellence, and its role as a regional hub.",
        "Who was Dedan Kimathi?",
        "Dedan Kimathi was the supreme commander of the Mau Mau Land and Freedom Army — the leading figure in the armed resistance against British colonial rule in Kenya during the 1950s. Born in 1920 in Nyeri, he became radicalised by the dispossession of Kikuyu land by white settlers and joined the independence movement.\n\nKimathi organised and commanded guerrilla operations from the forests of Mount Kenya and the Aberdare Range. The British, with support from colonial loyalists (Home Guards), launched a brutal counter-insurgency. Kimathi was captured in October 1956, tried for possessing a firearm, and hanged on 18 February 1957.\n\nHe is celebrated today as a national hero. His statue stands on Kimathi Street in Nairobi's CBD. His story is told in the play 'The Trial of Dedan Kimathi' by Ngugi wa Thiong'o and Micere Githae Mugo — one of Kenya's most important literary works."
    ),
]

# ── Category 5: Clarification & edge cases ────────────────────────────────────
CLARIFICATION_CONVOS = [
    conv(
        "Can you help me?",
        "Of course! I am here to help. What would you like assistance with today?",
        "I need information about something important.",
        "I would be happy to help. Could you tell me more about what you need? For example, is this about health, business, education, technology, or something else? The more specific you can be, the better I can assist you.",
        "It is about health.",
        "Great — health is an important topic. Are you looking for information about a specific condition or symptom, general wellness and prevention, nutrition and diet, mental health, or something else? Feel free to describe your situation and I will do my best to give you useful, clear information."
    ),
    conv(
        "I want to make a lot of money fast.",
        "I understand the desire for financial security — it is a completely normal goal. However, I want to be honest with you: most 'get rich fast' approaches are either scams, very high risk, or both. The strategies that actually work usually take consistent effort over time.",
        "What about forex trading? People say you can make millions.",
        "Forex trading can be profitable, but the reality is that research shows over 70% of retail forex traders lose money. It requires deep knowledge of global economics, technical analysis, strong emotional discipline, and significant starting capital. Many advertisements showing 'millionaire traders' are either fraudulent or showing survivorship bias — the rare successes, not the many failures. If you are interested in investing, starting with safer, lower-risk options (unit trusts, money market funds, NSE shares via a stockbroker) while you build knowledge is more likely to create lasting wealth.",
        "What is the safest way to grow money in Kenya?",
        "Here are tested, reliable options from lowest to higher risk:\n\n1. **Money Market Fund** (e.g., Sanlam, CIC, Britam) — returns 10-14% per year, accessible any time, low minimum (Ksh 1,000). This is the safest and best alternative to a savings account.\n2. **Government Bonds/T-Bills** — returns 11-16%, government-backed (very low risk), 91-day to 30-year maturities. Buy through CBK or your bank.\n3. **Saccos** — if you find a registered, well-managed Sacco, dividends and interest rebates can reach 10-15% per year, plus access to affordable loans.\n4. **NSE Stocks** — higher risk, higher potential returns, but requires research. Start with blue-chip companies (Safaricom, Equity, KCB).\n\nThe most powerful wealth-building tool available to everyone is compound interest — even small amounts invested consistently grow dramatically over time."
    ),
    conv(
        "What is the best phone to buy?",
        "That depends on your budget and what you need the phone for. Could you tell me: (1) What is your budget in Kenyan shillings? (2) What do you mainly use your phone for — calls and WhatsApp, social media, photography, mobile banking, or something more demanding?",
        "Budget around 15,000 shillings, mainly for WhatsApp, photos, and M-Pesa.",
        "For Ksh 15,000, you have several solid options available in Kenya. My top recommendations:\n\n**Tecno Camon 20** (around Ksh 14,000–16,000): 50MP camera, large battery (5000mAh), runs well on social media. Very popular in Kenya and widely available.\n\n**Infinix Hot 30** (around Ksh 12,000–15,000): Good display, strong battery, handles WhatsApp and camera well.\n\n**Samsung Galaxy A05** (around Ksh 13,000–16,000): Samsung reliability, decent camera, smooth performance. A bit more expensive but the Samsung brand tends to hold value and gets support longer.\n\nFor M-Pesa, any of these will work perfectly. I would lean toward the Tecno Camon 20 for the camera quality at that price point.",
        "Where is the best place to buy in Nairobi?",
        "For the best combination of genuine products and price, try: **Jumia** or **Kilimall** (online — compare prices, read reviews, but buy from verified sellers only). **Safaricom shops** or **Airtel stores** — guaranteed genuine, often with warranty. **Westgate Mall, Garden City, or Two Rivers** — authorised brand stores. Avoid buying phones from hawkers on the street or informal market stalls — counterfeit phones are common and you have no recourse if it fails. If buying from an independent shop in Westlands or Luthuli Avenue, test the phone thoroughly before paying."
    ),
]

# ── Category 6: Kenya-specific practical convos ───────────────────────────────
KENYA_PRACTICAL = [
    conv(
        "How do I apply for a passport in Kenya?",
        "To apply for a Kenyan passport, you can do it online or in person. The online process (recommended): (1) Create an account on eCitizen at ecitizen.go.ke. (2) Under 'Immigration Department,' select 'New Passport Application.' (3) Fill in the form — you'll need your National ID details. (4) Upload a recent passport photo (white background, 35x45mm). (5) Pay the fee online — Ksh 4,550 for ordinary passport (32 pages) or Ksh 4,550 for 64 pages. (6) Book an appointment at the Passport Office, Regional Office, or Huduma Centre. (7) Attend your appointment with originals of ID, birth certificate, and any other supporting documents.",
        "How long does it take to get it?",
        "Processing typically takes 10-21 working days for standard applications. Express processing (Ksh 5,050 extra) can reduce it to 3-7 working days. You can track your passport status on eCitizen. Once ready, you'll get an SMS notification and can collect it from the office where you applied — bring your ID and collection receipt.",
        "Can I apply from outside Nairobi?",
        "Yes. You can apply at Huduma Centres in all 47 county headquarters — Mombasa, Kisumu, Nakuru, Eldoret, Nyeri, etc. You can also submit applications at Kenya High Commissions and Embassies if you are abroad. The entire eCitizen application can be done from anywhere with internet access; you only need to appear in person once at your chosen collection point."
    ),
    conv(
        "What should I do if I feel sick in Kenya and cannot afford a private hospital?",
        "Kenya has a public health system that is accessible at low or no cost. Here is what to do: For minor illness, go to a Government Health Centre or Dispensary — these exist in every constituency and charge very little (Ksh 50-200 for consultation). For more serious illness, go to a Level 4 or Level 5 County Hospital (such as Kenyatta National Hospital in Nairobi, Coast General in Mombasa, JOOTRH in Kisumu). KNH is a national referral hospital that treats complex cases and has subsidised costs.",
        "What is NHIF and can it help?",
        "NHIF (National Hospital Insurance Fund) is Kenya's public health insurance. Any employed person earning over Ksh 1,000/month is required to contribute — between Ksh 150 and Ksh 1,700 per month depending on income. Self-employed and informal workers can contribute voluntarily at Ksh 500/month. NHIF covers inpatient treatment at accredited public and many private hospitals. It covers surgery, maternity, and chronic illness management. Sign up at any NHIF branch or through nhif.or.ke. Even Ksh 500/month gives you significant coverage.",
        "What about the SHA that replaced NHIF?",
        "The government introduced the Social Health Authority (SHA) in 2024 to replace NHIF. SHA is part of Universal Health Coverage (UHC) reforms. It uses three funds: (1) Primary Health Fund — covers outpatient care at public facilities, (2) Social Health Insurance Fund — covers inpatient and specialist care, (3) Emergency Fund — covers critical emergencies. Contributions are income-based: 2.75% of gross salary for formal employees. The transition has been gradual, so check the latest updates at sha.go.ke or at your nearest Huduma Centre."
    ),
    conv(
        "My child is about to start Standard 1. What should I prepare?",
        "Congratulations on this milestone! Here is what to prepare: **Documents needed for enrolment**: Birth certificate (must have), immunisation card (BCG, OPV, measles etc.), and transfer letter if moving from another school. **Items to buy**: School uniform (get the school's specific requirements), rubber shoes or bata (closed toe for safety), school bag (not too heavy), water bottle, lunchbox, exercise books (ruled), pencils, eraser, ruler, and colouring pencils.",
        "What about the CBC curriculum?",
        "The CBC (Competency Based Curriculum) replaced the old 8-4-4 system starting from 2017. Key differences: CBC focuses on practical skills and 'learning by doing' rather than memorisation for exams. Grades 1-3 (Lower Primary) use a portfolio and observation-based assessment, not exams. There are 7 core subjects at Lower Primary: Literacy (English/Kiswahili), Maths, Environmental Activities, Hygiene and Nutrition, Religious Education, Creative Activities, and Movement and Creative Arts. Learning is meant to be active and contextual — children make models, do projects, and learn through exploration.",
        "How can I support learning at home under CBC?",
        "Under CBC, home involvement is more important than ever. The curriculum explicitly expects parents to support projects and activities at home. Practical tips: Read with your child every evening (even 15 minutes) — literacy is the foundation for everything. Ask open-ended questions rather than yes/no ones ('What did you learn today?' not 'Was school good?'). Encourage curiosity — if they ask a question you cannot answer, look it up together. For projects: help them do it themselves rather than doing it for them. Attend school meetings and keep in touch with the class teacher."
    ),
    conv(
        "How does M-Pesa work for someone who has never used it?",
        "M-Pesa is a mobile money service run by Safaricom that lets you store, send, and receive money using your phone — no bank account needed. To get started: (1) Register at any Safaricom shop or authorised M-Pesa agent with your National ID and a Safaricom SIM. (2) Set a 4-digit PIN (keep this secret — never share it). (3) Load money ('deposit') at any M-Pesa agent — give them cash, they send you M-Pesa. Your balance is stored on your phone.",
        "How do I send money to someone?",
        "To send money: Dial *334# on your Safaricom phone → Select 'Send Money' → Enter the recipient's phone number → Enter amount → Enter your PIN → Confirm. The recipient gets an SMS instantly confirming they've received money. Charges are small — sending Ksh 100 costs Ksh 0; Ksh 1,000 costs Ksh 12; Ksh 10,000 costs Ksh 55. You can also send to any Kenyan mobile number (including Airtel and Telkom via the M-Pesa menu).",
        "What is Lipa na M-Pesa?",
        "Lipa na M-Pesa (Pay with M-Pesa) is how you pay businesses. There are two types: **Till Number (Buy Goods)** — used in shops, supermarkets, restaurants. You enter the till number, amount, and PIN. Money goes directly to the business. **Paybill** — used for bills like electricity (Kenya Power), water, rent, school fees, or bank deposits. You enter the business number (e.g., 888880 for Kenya Power), your account number (meter number or customer number), then amount and PIN. It is very secure — you always know exactly where your money is going."
    ),
]

# ── Category 7: Opinion / perspective questions ───────────────────────────────
OPINION_CONVOS = [
    conv(
        "Do you think technology is making life better or worse?",
        "Technology is making life better in many measurable ways and worse in a few important ones. On the better side: medicine has dramatically extended life expectancy and reduced suffering; communication tools allow families separated by distance to stay close; access to information and education has never been greater. M-Pesa alone has brought financial services to millions of Kenyans who had no bank account. On the worse side: social media is linked to increased anxiety and depression, especially in young people; technology is changing the job market faster than education systems can adapt; and the digital divide means technology's benefits are unevenly distributed.",
        "What do you think is the biggest technological challenge Kenya faces?",
        "The digital divide is Kenya's most pressing technology challenge. While Kenya leads Africa in mobile internet penetration, roughly 40% of the population still lacks regular internet access — mostly in rural and low-income areas. Without access, people miss out on e-government services, digital banking, online education, and the remote work economy. The solution requires both infrastructure (cheaper data, wider coverage) and digital literacy programmes so people know how to use the tools available to them.",
        "Is AI a threat to jobs in Kenya?",
        "AI will displace certain jobs and create others — the net effect in Kenya depends on how quickly the education system adapts. Jobs most at risk: data entry, basic customer service, routine accounting tasks, and repetitive manufacturing. Jobs that AI will create or expand: AI trainers and auditors, data labellers (Kenya already has a growing data annotation industry), software developers, and roles requiring human judgement and relationships. The realistic near-term picture for Kenya is that AI acts as a productivity tool for workers rather than a direct replacement — most Kenyan jobs require physical presence, relationship skills, or local context that AI cannot replicate yet."
    ),
    conv(
        "What are the advantages and disadvantages of living in a city versus the countryside in Kenya?",
        "Both have genuine advantages depending on your priorities.\n\n**City (Nairobi/Mombasa)**: More jobs and career opportunities; better hospitals and specialist care; more schools and universities; entertainment and social life; faster internet and infrastructure. Disadvantages: high cost of living, traffic, pollution, less community cohesion, and higher stress.\n\n**Countryside**: Lower cost of living (especially housing and food); strong community and family networks; cleaner air and environment; land ownership opportunities; and for many Kenyans, a stronger sense of belonging and identity. Disadvantages: fewer formal jobs, longer distances to specialist health services, and sometimes limited educational options.",
        "Where would you recommend for raising children?",
        "I think the countryside has underrated advantages for raising children: more space, more natural play, stronger extended family presence, and lower exposure to urban crime and the pressures of status competition that cities can amplify. However, access to quality schools is crucial, and if the local schools are significantly weaker, the urban advantage in education may outweigh the other factors. A practical option many Kenyan families choose: one parent works in the city while children and the other parent stay upcountry, balancing income with a healthier environment. Technology is making this more viable as remote work grows."
    ),
]

# ── Combine all and write ─────────────────────────────────────────────────────
ALL = []
for category in [SCIENCE_CONVOS, HOWTO_CONVOS, SUPPORT_CONVOS,
                 FORMAT_CONVOS, CLARIFICATION_CONVOS, KENYA_PRACTICAL,
                 OPINION_CONVOS]:
    # Each conversation is high-value — include 4x for strong signal
    ALL.extend(category * 4)

random.shuffle(ALL)

os.makedirs(os.path.join(REPO_ROOT, "data/sft"), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for item in ALL:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

total_turns = sum(len(x["messages"]) - 1 for x in ALL)  # -1 for system
print(f"Generated {len(ALL)} conversations → {OUT_FILE}")
print(f"Total turns (user+assistant): {total_turns}")
print(f"  Science:         {len(SCIENCE_CONVOS)} x4 = {len(SCIENCE_CONVOS)*4}")
print(f"  How-to:          {len(HOWTO_CONVOS)} x4 = {len(HOWTO_CONVOS)*4}")
print(f"  Emotional:       {len(SUPPORT_CONVOS)} x4 = {len(SUPPORT_CONVOS)*4}")
print(f"  Format-follow:   {len(FORMAT_CONVOS)} x4 = {len(FORMAT_CONVOS)*4}")
print(f"  Clarification:   {len(CLARIFICATION_CONVOS)} x4 = {len(CLARIFICATION_CONVOS)*4}")
print(f"  Kenya practical: {len(KENYA_PRACTICAL)} x4 = {len(KENYA_PRACTICAL)*4}")
print(f"  Opinion:         {len(OPINION_CONVOS)} x4 = {len(OPINION_CONVOS)*4}")
