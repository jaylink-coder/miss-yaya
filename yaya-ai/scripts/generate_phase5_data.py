import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILE = os.path.join(REPO_ROOT, "data/sft/yaya_phase5_knowledge.jsonl")

SYS = "You are Yaya, a helpful and knowledgeable AI assistant. You explain things clearly and accurately, giving enough detail to be genuinely useful."

def qa(q, a, system=SYS):
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}

# --- Human Biology ---
BIOLOGY = [
    qa(
        "How does digestion work?",
        "Digestion is the process of breaking food down into nutrients the body can absorb. It starts in the mouth, where chewing physically breaks food apart and saliva begins chemically breaking down starches. Food travels down the oesophagus to the stomach, where acid and enzymes break down proteins. The partially digested food (called chyme) then moves into the small intestine, where most nutrient absorption happens — aided by bile from the liver and enzymes from the pancreas. The large intestine absorbs water and forms solid waste, which is excreted. The whole process takes 24–72 hours. Nutrients absorbed in the small intestine pass into the bloodstream and are transported to cells throughout the body for energy, growth, and repair."
    ),
    qa(
        "How does the human heart work?",
        "The heart is a muscular pump that circulates blood throughout the body. It has four chambers: two upper chambers called atria (which receive blood) and two lower chambers called ventricles (which pump blood out). The right side of the heart receives oxygen-depleted blood from the body and sends it to the lungs, where it picks up oxygen. The left side receives oxygen-rich blood from the lungs and pumps it out to the rest of the body. Heart valves open and close to keep blood flowing in one direction. The 'lub-dub' sound you hear is those valves closing. A healthy adult heart beats about 60–100 times per minute and pumps around 5 litres of blood per minute — enough to circle the Earth twice in a lifetime."
    ),
    qa(
        "What are blood types and why do they matter?",
        "Blood types are classifications based on proteins (antigens) on the surface of red blood cells. The ABO system gives us four main types: A, B, AB, and O. Type A has A antigens, type B has B antigens, type AB has both, and type O has neither. There is also the Rh factor — you are either positive (+) or negative (-), giving 8 possible blood types in total. Blood types matter most during transfusions. If you receive the wrong blood type, your immune system attacks the donated blood, which can be fatal. Type O-negative is called the 'universal donor' because it lacks both ABO and Rh antigens. Type AB-positive is the 'universal recipient.' Knowing your blood type is important for emergencies, surgery, and organ transplants."
    ),
    qa(
        "How does the immune system protect the body?",
        "The immune system is the body's defence network against disease. It has two main parts. The innate immune system is the first line of defence — skin, mucus, and white blood cells (like neutrophils) that attack any foreign invader immediately. The adaptive immune system is slower but more targeted. When a new pathogen enters the body, specialised white blood cells called B-cells produce antibodies — proteins that bind to the specific invader and mark it for destruction. T-cells help coordinate this response and kill infected cells directly. Crucially, the adaptive system has memory: after fighting an infection once, it remembers the pathogen and can respond much faster if it appears again — this is why you don't get chickenpox twice, and why vaccines work."
    ),
    qa(
        "What is the difference between arteries and veins?",
        "Arteries and veins are both blood vessels, but they carry blood in different directions and have different structures. Arteries carry blood away from the heart — usually oxygen-rich blood — under high pressure. Because of this, they have thick, muscular, elastic walls to handle the force. Veins carry blood back to the heart — usually oxygen-depleted — under much lower pressure. Their walls are thinner, and they have valves to prevent blood flowing backwards. The exception is the pulmonary system: the pulmonary artery carries deoxygenated blood from the heart to the lungs, and the pulmonary veins carry oxygenated blood back. Capillaries are the tiny vessels that connect arteries to veins, and it is here that the actual exchange of oxygen and nutrients with body tissues takes place."
    ),
]

# --- Earth Science ---
EARTH = [
    qa(
        "Why do seasons change throughout the year?",
        "Seasons change because Earth is tilted on its axis by about 23.5 degrees — not because it moves closer to or farther from the Sun (a common misconception). As Earth orbits the Sun, different hemispheres tilt toward or away from the Sun at different times of year. When the Northern Hemisphere tilts toward the Sun (June–August), it receives more direct sunlight for longer each day, creating summer. At the same time, the Southern Hemisphere tilts away, creating winter. Six months later, the situation reverses. Equatorial countries like Kenya experience less dramatic seasons because the sun angle and day length stay relatively consistent year-round. Instead, Kenya has wet and dry seasons driven by the movement of the Inter-Tropical Convergence Zone (ITCZ), not Earth's axial tilt."
    ),
    qa(
        "How do volcanoes form?",
        "Volcanoes form where molten rock (magma) from deep inside the Earth pushes up to the surface. This happens in three main settings. At tectonic plate boundaries, where two plates diverge (pull apart), magma wells up to fill the gap — this creates underwater ridges and volcanic islands. Where plates converge (collide), one plate sinks below the other (subduction), melting as it goes and generating magma that rises explosively — this creates the famous 'Ring of Fire' volcanoes around the Pacific. The third type forms over 'hot spots' — unusually hot regions of the mantle that burn through the plate above. Hawaii's islands were formed this way as the Pacific Plate slowly moved over a stationary hot spot. The East African Rift Valley, which passes through Kenya, is a zone where the African continent is slowly splitting apart, creating volcanic activity including Mount Kenya and the Rift Valley lakes."
    ),
    qa(
        "What is continental drift and is it still happening?",
        "Continental drift is the theory that Earth's continents were once joined together in a single supercontinent (called Pangaea about 300 million years ago) and have been slowly moving apart ever since. The evidence is compelling: the coastlines of Africa and South America fit together like puzzle pieces; identical fossils are found on continents now separated by thousands of kilometres; and mountain ranges on different continents align when you put them back together. This movement is driven by plate tectonics — the slow convection of hot rock in the Earth's mantle pushes the rigid plates of the crust around. Yes, it is still happening today. Africa and the Arabian Peninsula are moving apart, widening the Red Sea. The East African Rift is slowly splitting eastern Africa away from the rest of the continent — in tens of millions of years, a new ocean could form where Kenya stands today."
    ),
    qa(
        "How does rainfall form?",
        "Rainfall forms through the water cycle. The sun heats water in oceans, lakes, and rivers, causing it to evaporate into water vapour. Plants also release water vapour through a process called transpiration. This warm, moist air rises into the atmosphere, where it cools. Cool air can hold less water vapour than warm air, so the vapour condenses onto tiny particles (dust, pollen) to form droplets — this creates clouds. When the droplets combine and grow heavy enough, they fall as rain. In colder conditions, they fall as snow or hail. In Kenya, rainfall patterns are closely tied to the movement of the ITCZ — the zone where the northeast and southeast trade winds meet. When it passes over Kenya, it brings the long rains (March–May) and short rains (October–December). Coastal areas and highland regions like the Aberdares receive more rainfall due to orographic lift — air cools as it rises over mountains."
    ),
    qa(
        "What causes earthquakes?",
        "Earthquakes are caused by the sudden release of energy stored in Earth's crust. The crust is broken into tectonic plates that are constantly moving — very slowly, a few centimetres per year. At fault lines, where plates meet, friction locks them together even as the underlying forces continue pushing. Stress builds up over decades or centuries until the rocks suddenly slip, releasing enormous energy as seismic waves that travel through the earth and shake the surface. The point underground where the slip occurs is the focus (or hypocenter); the point on the surface directly above is the epicentre. The Richter scale measures energy release: each whole number increase represents about 32 times more energy. A magnitude 7.0 earthquake releases 32 times more energy than a 6.0. Kenya sits near the East African Rift, which produces minor to moderate earthquakes, particularly around Naivasha, Magadi, and the Lake Victoria region."
    ),
]

# --- Health & Medicine ---
HEALTH = [
    qa(
        "How do vaccines work?",
        "Vaccines work by training your immune system to recognise and fight a specific pathogen without you having to suffer the actual disease. A vaccine introduces your body to something that looks like the pathogen — it could be a weakened or killed version of the germ, a harmless piece of its protein coat, or (with newer mRNA vaccines like COVID-19 vaccines) instructions for your cells to temporarily produce a protein that the immune system can recognise. Your immune system responds as if under real attack: it creates antibodies and generates 'memory cells' that remember the threat. If you later encounter the real pathogen, your immune system recognises it immediately and responds so fast the infection is defeated before you become seriously ill. Vaccines don't just protect individuals — when enough people are vaccinated, 'herd immunity' protects even those who cannot be vaccinated, such as newborns or immunocompromised people."
    ),
    qa(
        "What is diabetes and how does it affect the body?",
        "Diabetes is a condition where the body cannot properly regulate blood sugar (glucose) levels. Normally, when you eat, blood glucose rises and the pancreas releases insulin — a hormone that allows cells to absorb glucose for energy. In Type 1 diabetes, the immune system destroys the insulin-producing cells in the pancreas, so the body makes little or no insulin. People with Type 1 must take insulin injections to survive. In Type 2 diabetes (far more common), the body becomes resistant to insulin over time, often linked to diet, inactivity, and genetics. Without proper insulin function, glucose builds up in the blood. Over years, high blood sugar damages blood vessels and nerves, causing serious complications: kidney failure, blindness, heart disease, nerve damage in the feet (which can lead to amputation), and poor wound healing. Type 2 diabetes can often be managed or even reversed with diet changes, exercise, and weight loss."
    ),
    qa(
        "How can you prevent malaria?",
        "Malaria is caused by Plasmodium parasites transmitted through the bites of infected female Anopheles mosquitoes. Prevention focuses on avoiding those bites. Sleeping under insecticide-treated mosquito nets (ITNs) is one of the most effective and affordable protections — Kenya's government distributes these widely. Applying insect repellent (containing DEET) on exposed skin also helps. Wearing long-sleeved clothing and trousers in the evenings reduces exposure. Indoors, insecticide sprays and plug-in repellents reduce mosquito numbers. Eliminating standing water around your home removes mosquito breeding sites. For travellers entering high-risk zones, prophylactic (preventive) antimalarial medications are recommended. In 2021, the WHO approved the RTS,S malaria vaccine — the first licensed malaria vaccine — which Kenya rolled out in its national immunisation programme. Early treatment is equally important: if you develop fever, chills, or headache in a malaria zone, seek a blood test immediately."
    ),
    qa(
        "What are the facts about HIV/AIDS?",
        "HIV (Human Immunodeficiency Virus) attacks the immune system, specifically CD4 T-cells, which coordinate the body's defence against infection. Without treatment, HIV gradually destroys enough CD4 cells that the immune system can no longer fight infections — this stage is called AIDS (Acquired Immunodeficiency Syndrome). HIV is transmitted through specific body fluids: blood, semen, vaginal fluids, and breast milk. It is NOT transmitted through casual contact, sharing food, handshakes, or mosquito bites. Prevention includes consistent condom use, not sharing needles, and PrEP (pre-exposure prophylaxis) medication for high-risk individuals. Crucially, HIV today is a manageable chronic condition with antiretroviral therapy (ART). People on ART can live long, healthy lives and, when the viral load is undetectable, cannot transmit the virus sexually (U=U: Undetectable = Untransmittable). Kenya has made significant progress — millions of Kenyans are on ART through programmes supported by the government and international partners."
    ),
    qa(
        "What is cholesterol and when is it dangerous?",
        "Cholesterol is a fatty substance found in the blood that the body needs to build cells, make hormones, and produce vitamin D. Your liver makes all the cholesterol your body needs, but you also get it from animal products like meat, dairy, and eggs. There are two main types: LDL (low-density lipoprotein) — often called 'bad' cholesterol — deposits cholesterol on artery walls, forming plaques that narrow and harden arteries (atherosclerosis). HDL (high-density lipoprotein) — 'good' cholesterol — carries cholesterol back to the liver for disposal. High LDL is dangerous because it increases the risk of heart attack and stroke. Narrowed arteries restrict blood flow; if a clot forms in a narrowed artery, blood flow can stop entirely. Risk factors for high LDL include a diet high in saturated fats, smoking, lack of exercise, obesity, diabetes, and genetics. Treatment includes dietary changes, exercise, and statin medications."
    ),
]

# --- Technology ---
TECH = [
    qa(
        "How does WiFi work?",
        "WiFi is a wireless networking technology that uses radio waves to transmit data between devices and a router. Your router is connected to the internet via a physical cable (from your ISP). The router converts the incoming data into radio signals and broadcasts them in all directions. Your phone, laptop, or other device has a wireless adapter that receives these signals and converts them back into data. Communication goes both ways: when you send data (e.g., loading a webpage), your device broadcasts radio signals back to the router. WiFi typically operates on two frequency bands: 2.4 GHz (longer range, slower) and 5 GHz (shorter range, faster). The 'GHz' refers to the frequency of the radio waves used. Data is broken into small packets, sent across the network, and reassembled at the destination. Security comes from encryption (WPA2/WPA3) — without it, anyone within range could intercept your data."
    ),
    qa(
        "What is 'the cloud' in computing?",
        "The cloud refers to servers located in data centres around the world that you access over the internet, rather than storing and processing data on your own device. When you use Google Drive, Dropbox, or iCloud, your files are not stored on your laptop — they are stored on remote servers maintained by those companies. When you stream music on Spotify, the songs aren't on your phone — they stream from servers in real time. The advantages are significant: you can access your data from any device, anywhere; you don't need expensive local storage; and providers manage security, backups, and maintenance. For businesses, the cloud means they can rent computing power (from providers like AWS, Google Cloud, or Microsoft Azure) and scale up or down as needed, without buying expensive hardware. In Kenya, cloud services have enabled businesses to operate sophisticated software without large upfront infrastructure costs."
    ),
    qa(
        "How does GPS know where you are?",
        "GPS (Global Positioning System) works by measuring signals from satellites. The US government operates 31 GPS satellites orbiting Earth at about 20,200 km altitude. Each satellite continuously broadcasts a signal containing its exact position and the current time. Your GPS receiver listens to these signals. By calculating how long each signal took to arrive (and therefore how far away each satellite is), your device can use triangulation to pinpoint your location. To get an accurate 3D position (latitude, longitude, and altitude), you need signals from at least four satellites. The more satellites visible, the more accurate the reading. GPS accuracy is typically 3–5 metres for civilian devices. Modern smartphones also use assisted GPS (A-GPS), which uses cellular towers and WiFi hotspot locations to speed up the initial fix and improve accuracy — this is why your maps app can find you faster when you have mobile data."
    ),
    qa(
        "What is an algorithm?",
        "An algorithm is a step-by-step set of instructions for solving a problem or completing a task. Think of it like a recipe: you start with inputs (ingredients), follow specific steps in order, and end with an output (a dish). Algorithms are the foundation of all computer software — every app, search engine, and social media feed runs on algorithms. A simple example is a sorting algorithm: given a list of names in random order, an algorithm can arrange them alphabetically by comparing pairs and swapping them until the list is sorted. More complex algorithms power things like Google Search (which ranks billions of webpages in milliseconds), recommendation systems (which decide what videos YouTube suggests next), and navigation apps (which calculate the fastest route through thousands of roads). The quality of an algorithm matters enormously: a well-designed algorithm can solve a problem millions of times faster than a poorly designed one."
    ),
    qa(
        "How does mobile money like M-Pesa work?",
        "M-Pesa is a mobile money transfer system that lets people send and receive money using a basic mobile phone — no bank account required. When you deposit cash at an M-Pesa agent, the equivalent value is added to your electronic wallet, stored in Safaricom's system. When you send money to another number, Safaricom deducts from your wallet and credits the recipient's wallet instantly. The recipient can withdraw cash at any M-Pesa agent by showing an SMS code. The security model relies on PIN authentication and transaction SMS confirmations. Behind the scenes, Safaricom holds all customer funds in regulated trust accounts at commercial banks, ensuring the system is backed by real money. M-Pesa has been transformative in Kenya because it gave financial access to millions who couldn't open bank accounts due to lack of documentation, distance, or minimum balance requirements. It launched in 2007 and is now used by over 30 million Kenyans."
    ),
]

# --- Business & Economics ---
ECONOMICS = [
    qa(
        "What is inflation and what causes it?",
        "Inflation is the rate at which the general price level of goods and services rises over time, which means the purchasing power of money falls. If inflation is 10%, something that cost 100 shillings last year costs 110 shillings today. There are two main causes. Demand-pull inflation occurs when too much money chases too few goods — for example, when government spending or consumer borrowing increases rapidly. Cost-push inflation occurs when the cost of production rises — for example, when fuel prices spike (making transport and manufacturing more expensive) or when droughts reduce food supply. Mild inflation (2–5%) is generally considered healthy in a growing economy. Hyperinflation — like Zimbabwe experienced in 2008, when prices doubled every day — is devastating, wiping out savings and destabilising the economy. Central banks like the Central Bank of Kenya manage inflation by raising interest rates (making borrowing more expensive, cooling spending) or cutting rates (stimulating growth)."
    ),
    qa(
        "How do supply and demand work?",
        "Supply and demand is the fundamental model explaining how prices are set in a free market. Demand refers to how much of a product consumers want to buy at a given price — generally, the lower the price, the more people want to buy it. Supply refers to how much producers are willing to sell at a given price — generally, the higher the price, the more they want to produce. The price settles at the point where supply and demand balance — called the equilibrium. When demand rises faster than supply (say, avocado becomes popular globally), prices go up. When supply exceeds demand (a bumper harvest floods the market with maize), prices fall. This is why matatu fares go up during peak hours — demand for rides is high and supply (number of matatus) is fixed. Understanding supply and demand helps farmers, businesses, and governments make better decisions about what to produce and when to sell."
    ),
    qa(
        "How do stock markets work?",
        "A stock market is a place where shares (small ownership stakes) in companies are bought and sold. When a company wants to raise money to grow, it can 'go public' by issuing shares through an Initial Public Offering (IPO). Investors buy these shares, becoming part-owners of the company. If the company does well, the shares become more valuable and investors profit. If it struggles, the share price falls. Investors can also earn dividends — a share of profits paid out regularly. Stock prices fluctuate based on supply and demand for the shares, which is driven by company performance, economic conditions, and investor sentiment. The Nairobi Securities Exchange (NSE) is Kenya's stock market, where companies like Safaricom, Equity Bank, and KCB are listed. You can invest through a stockbroker or through apps like Hisa that make fractional investing accessible. Stock markets also serve as economic indicators — falling markets often signal declining confidence in the economy."
    ),
    qa(
        "What does GDP mean and why does it matter?",
        "GDP stands for Gross Domestic Product — the total value of all goods and services produced within a country in a given period, usually a year. It is the most widely used measure of the size of an economy. If Kenya's GDP is $100 billion, that means the total output of the Kenyan economy in that year was worth $100 billion. GDP growth means the economy is expanding — more goods are being produced, more services rendered, more employment generated. GDP per capita (GDP divided by population) gives a rough measure of average living standards. However, GDP has limitations: it measures output, not wellbeing. A country can have high GDP while having extreme inequality, environmental degradation, or poor social services. GDP also doesn't capture unpaid work like childcare or subsistence farming, which are hugely important in Kenya. Alternative measures like the Human Development Index (HDI) try to capture health and education alongside income."
    ),
]

# --- Kenya Specific ---
KENYA = [
    qa(
        "How does the Kenyan court system work?",
        "Kenya's court system is structured in a hierarchy. At the top is the Supreme Court, which is the highest court in the land. It hears appeals on constitutional matters and presidential election petitions. Below it is the Court of Appeal, which handles appeals from the High Court. The High Court deals with serious criminal cases, major civil cases, and constitutional matters — it also supervises the courts below it. Magistrates' Courts handle the majority of criminal cases and civil disputes involving smaller amounts. There are also specialised courts: the Employment and Labour Relations Court, the Environment and Land Court, the Anti-Corruption Court, and Kadhi Courts (for Muslim personal law matters). The judiciary is supposed to be independent of the executive and legislature — a principle enshrined in Kenya's 2010 Constitution, which significantly strengthened judicial independence after decades of executive interference."
    ),
    qa(
        "What is devolution in Kenya and what has it achieved?",
        "Devolution, introduced in Kenya's 2010 Constitution, shifted significant government powers and resources from the national government to 47 newly created county governments. Before devolution, nearly all public services and spending were controlled from Nairobi, meaning remote areas were chronically neglected. Now each county has a governor, a county assembly, and a budget funded partly by an equitable share of national revenue (currently around 15% of national revenues) and partly by local taxes and grants. Achievements include improved access to health services in rural areas (through hiring county health workers and building dispensaries), better local infrastructure (roads, markets), and increased political participation at the local level. Critics point to corruption in county governments, duplication of roles, and uneven service delivery. Overall, devolution has been considered a net positive — the biggest constitutional change in Kenya's modern history — though its potential is still being realised."
    ),
    qa(
        "How does Safaricom's business model work?",
        "Safaricom is Kenya's largest telecommunications company and one of Africa's most profitable. Its revenue comes from several streams. Voice calls and SMS were historically the core, though these have declined as data usage grows. Mobile data is now a major revenue driver — Safaricom operates 4G and 5G networks. M-Pesa is Safaricom's crown jewel: it charges transaction fees every time money is sent, withdrawn, or used to pay a bill. These fees are small individually but massive in aggregate — M-Pesa processes hundreds of billions of shillings monthly. Safaricom also earns from enterprise services (providing IT and communication solutions to businesses), selling devices, and content services. The company has a dominant market position (over 60% market share) that gives it pricing power. It is listed on the Nairobi Securities Exchange and is partly owned by Vodacom, Vodafone, and the Kenyan government. Safaricom's success story is studied globally as a model for how telecoms can drive financial inclusion."
    ),
]

# --- Agriculture ---
AGRICULTURE = [
    qa(
        "What are the main soil types and how do they affect farming?",
        "Soil is broadly categorised into several types based on texture — the mix of sand, silt, and clay particles. Sandy soil has large particles with big spaces between them, draining water very quickly. This means it doesn't retain moisture or nutrients well — not ideal for most crops without irrigation and compost. Clay soil has very fine particles that pack tightly, retaining water but draining poorly. It can become waterlogged, starving roots of oxygen, and sets rock-hard when dry. Loam soil is the ideal mix of sand, silt, and clay — it retains moisture and nutrients while still draining well. Most productive farmland is loamy. In Kenya, soil types vary dramatically: the highland areas around Mount Kenya and the Aberdares have fertile volcanic and loamy soils excellent for tea, coffee, and maize. Coastal soils tend to be sandy. The Rift Valley floor has alkaline or saline soils in some areas. Farmers can improve any soil by adding organic matter (compost, manure), which improves both drainage in clay and water retention in sandy soils."
    ),
    qa(
        "Why is crop rotation important?",
        "Crop rotation means growing different crops on the same land in successive seasons, rather than the same crop year after year. It is one of the most important and ancient farming practices. The main benefits are: First, different crops deplete different soil nutrients. Growing the same crop repeatedly exhausts specific nutrients — maize is a heavy feeder of nitrogen, for example. Rotating with legumes (beans, peas, groundnuts) restores nitrogen naturally because legumes have bacteria on their roots that fix nitrogen from the air into the soil. Second, rotation breaks pest and disease cycles. Many pests and soil-borne diseases are crop-specific. If you keep planting maize, pests that feed on maize build up in the soil. Switch to a different crop, and those pests decline for lack of food. Third, different root depths affect soil structure differently, reducing compaction. For smallholder farmers in Kenya, a simple rotation of maize → beans → vegetables can significantly improve yields and reduce the need for expensive fertiliser."
    ),
    qa(
        "What are the main irrigation methods used in farming?",
        "Irrigation is the artificial supply of water to crops. There are several methods, each suited to different contexts. Surface irrigation (flood or furrow) involves channelling water along the ground between crop rows. It is cheap and simple but inefficient — up to 60% of water can be lost to evaporation or runoff. Sprinkler irrigation mimics rainfall using pipes, pumps, and spray heads. It is more efficient and better for uneven terrain, but requires more infrastructure and energy. Drip irrigation delivers water directly to the base of each plant through a network of pipes and emitters. It is the most water-efficient method (up to 90% efficient) because water goes directly to roots with minimal evaporation. However, it is more expensive to install. In Kenya, smallholder farmers increasingly use low-cost drip kits, which use gravity from an elevated tank — no pump needed. Large-scale irrigation schemes exist in the Tana River basin, Mwea (famous for rice), and around Naivasha (flower farms)."
    ),
    qa(
        "What is the importance of cover crops?",
        "Cover crops are plants grown primarily to benefit the soil rather than to be harvested. Farmers plant them during off-seasons or between main crop rows. Their benefits are numerous. They protect the soil surface from erosion — bare soil is vulnerable to being washed away by rain (especially important on sloped land) or blown away by wind. Their roots break up compacted soil, improving aeration and water infiltration. When the cover crop is cut and left on the surface (as mulch) or dug in (as green manure), it adds organic matter, improving soil structure and feeding soil microorganisms. Leguminous cover crops like mucuna (velvet bean), lablab, or cowpea also fix atmospheric nitrogen, reducing the need for synthetic fertiliser. In Kenya's smallholder context, cover crops have been promoted as part of conservation agriculture, which also emphasises minimum tillage and mulching — an approach that reduces costs while improving long-term soil health and resilience to drought."
    ),
]

ALL = []
for category, mult in [
    (BIOLOGY, 5),
    (EARTH, 5),
    (HEALTH, 5),
    (TECH, 5),
    (ECONOMICS, 5),
    (KENYA, 6),
    (AGRICULTURE, 5),
]:
    ALL.extend(category * mult)

random.shuffle(ALL)
os.makedirs(os.path.join(REPO_ROOT, "data/sft"), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for item in ALL:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Generated {len(ALL)} examples -> {OUT_FILE}")
