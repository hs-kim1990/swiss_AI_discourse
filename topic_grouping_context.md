# Swiss AI Discourse — Topic Grouping Context

## Dataset

- **Corpus**: German-language Swiss news articles (`German_politic.tsv`)
- **Classification file**: `results/German_politic_20260503_194531.json`
- **Total documents**: 78,851
- **NOT TOPIC** (irrelevant to initiative): 38,781 (49.2%)
- **Topic-relevant articles**: 40,070 (50.8%)
- **Classification field**: `classification` in each record

## Context

Articles were LLM-classified into ~2,500 granular subtopics, then consolidated into
**10 major categories** and further grouped into thematic **subgroups**.
The Swiss Domestic Politics category was audited and 775 misclassified articles
were migrated to more specific categories (confirmed via title-keyword analysis).

## Classification Values

The `classification` field in each JSON record takes one of these exact string values:

| Value | Articles | % of topic |
|-------|----------|------------|
| `NOT TOPIC` | 38,781 | — |
| `Swiss Domestic Politics & Governance` | 10,981 | 27.4% |
| `Economy, Labor & Finance` | 6,514 | 16.3% |
| `Immigration & Asylum Policy` | 5,751 | 14.4% |
| `Swiss-EU Relations & Bilateral Agreements` | 5,111 | 12.8% |
| `Refugee & Migrant Integration` | 4,890 | 12.2% |
| `Defense, Security & Military` | 2,251 | 5.6% |
| `Environment, Energy & Infrastructure` | 1,586 | 4.0% |
| `Healthcare, Social Welfare & Education` | 1,411 | 3.5% |
| `International Relations & Geopolitics` | 1,283 | 3.2% |
| `Demographic Change & Population` | 292 | 0.7% |

## Major Categories — Descriptions & Subgroups

### Swiss Domestic Politics & Governance
**Articles**: 10,981 (27.4% of topic corpus)

**Description**: Swiss party and electoral politics (especially SVP), government institutions, Bundesrat, direct democracy, media policy, civil rights, culture, identity, and local governance. Represents the broad political landscape in which the initiative was debated.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Government & Parliamentary Affairs | 3,986 | 36.3% | Swiss Politics (2127), Government Accountability (323), Swiss Local Politics (141) |
| Media, Transparency & Accountability | 2,572 | 23.4% | Transparency (2035), Swiss Media Policy (187), Transparency (80) |
| Local & Regional Governance | 2,450 | 22.3% | Animal Welfare (65), Housing Crisis (27), Swiss Municipal Leadership (23) |
| Culture, Society & Identity | 1,827 | 16.6% | Art (1137), Art (79), Swiss History (50) |
| Civil Rights, Equality & Justice | 790 | 7.2% | Racism Prevention (54), Antisemitismus (50), Swiss Disability Rights (46) |
| Elections & Party Politics | 131 | 1.2% | Swiss Election Policy (17), Swiss Right-Wing Extremism (17), Swiss Green Party Leadership (14) |

### Economy, Labor & Finance
**Articles**: 6,514 (16.3% of topic corpus)

**Description**: Banking, taxation, housing and urban development, labor market, trade and international commerce, agriculture and industry, social benefits and pensions. Covers economic arguments for and against immigration limits.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Banking, Finance & Taxation | 1,989 | 30.5% | Swiss Pension Reform (223), Swiss Tax Policy (207), Swiss Banking Policy (159) |
| Housing & Urban Development | 1,148 | 17.6% | Swiss Housing Policy (255), Swiss Infrastructure Developme (184), Swiss Urban Planning (98) |
| Other Economic Topics | 1,137 | 17.5% | Swiss Economy (98), Media Regulation (78), Swiss Military Budget (70) |
| Agriculture, Industry & Business | 694 | 10.7% | Swiss Agriculture Policy (98), Swiss Industrial Policy (78), Swiss Animal Welfare Policy (31) |
| Trade & International Commerce | 654 | 10.0% | Swiss US Trade Relations (135), Swiss US Relations (112), Trade Agreement (97) |
| Labor Market & Wages | 480 | 7.4% | Arbeitsmarkt in der Schweiz (141), Swiss Labor Market (66), Swiss Labor Market Reform (58) |
| Social Benefits, Pensions & Welfare | 127 | 1.9% | Social Welfare (33), Swiss Pension Policy (28), Swiss Social Welfare Policy (27) |

### Immigration & Asylum Policy
**Articles**: 5,751 (14.4% of topic corpus)

**Description**: Asylum procedures and reform, Swiss migration policy, migration control and border management, refugee housing and facilities, EU and international asylum frameworks. Core policy domain of the initiative.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Asylum Procedures & Reform | 2,480 | 43.1% | Swiss Asylum Policy (1332), Asylum Policy (239), Asylum Policy Reform (72) |
| Swiss Migration Policy | 2,168 | 37.7% | Swiss Migration Policy (1211), Swiss Immigration Policy (669), Swiss Migration (84) |
| EU & International Asylum Policy | 456 | 7.9% | Swiss EU Migration Policy (150), US Immigration Policy (106), Swiss Ukraine Refugee Policy (65) |
| Migration Control & Border Policy | 359 | 6.2% | Swiss Migration Crime Preventi (70), Swiss Migration Crime (54), Swiss Migration Crisis (36) |
| Other Migration Topics | 148 | 2.6% | Swiss Refugee Policy (45), Swiss Gaza Refugee Policy (12), Immigration Policy (10) |
| Refugee Housing & Facilities | 15 | 0.3% | Refugee Housing (6), Ausschaffungsgefängnis (4), Refugee Housing (4) |

### Swiss-EU Relations & Bilateral Agreements
**Articles**: 5,111 (12.8% of topic corpus)

**Description**: Core Switzerland-EU framework (bilateral agreements, free movement of persons), Brexit, EU trade and economic cooperation, EU security and defense, EU institutional affairs. Critical because the initiative threatened the bilateral path.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Core Switzerland-EU Framework | 4,236 | 82.9% | Swiss EU Relations (3707), EU Relations (243), EU Relations (90) |
| Brexit & European Realignment | 530 | 10.4% | Brexit (516), Brexit (9), Brexit Politics (3) |
| EU Trade & Economic Cooperation | 225 | 4.4% | Swiss EU Trade Policy (141), Swiss EU Trade Negotiations (72), Swiss EU Labor Agreement (8) |
| EU Security & Defense | 47 | 0.9% | Swiss EU Sanctions Policy (31), EU-Russia Relations (7), EU Security Policy (4) |
| EU Institutional Affairs | 36 | 0.7% | EU Commissioners (19), Swiss EU Politics (6), EU Commissioners (3) |
| Other EU Topics | 4 | 0.1% | Swiss EU Transgender Rights (2), Swiss EU Regulatory Compliance (1), EU-Ukraine Relations (1) |

### Refugee & Migrant Integration
**Articles**: 4,890 (12.2% of topic corpus)

**Description**: Refugee experiences and stories, integration programs and policy, employment and economic integration, cultural and social integration, education and youth integration.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Refugee Experiences & Stories | 4,257 | 87.1% | Refugee Experiences (4257) |
| Integration Programs & Policy | 313 | 6.4% | Swiss Integration Policy (74), Swiss Refugee Integration Poli (50), Swiss Integration Challenges (41) |
| Other Integration Topics | 138 | 2.8% | Integration (22), Integration through Immigratio (14), European Integration (13) |
| Cultural & Social Integration | 105 | 2.1% | Youth Integration (13), Integration through Social Ser (9), Integration through Housing (9) |
| Employment & Economic Integration | 46 | 0.9% | Swiss Economic Integration (16), Integration through Employment (11), Economic Integration (7) |
| Education & Youth Integration | 31 | 0.6% | Integration through Education (20), Integration through Education (10), Integration of Young Afghan Mi (1) |

### Defense, Security & Military
**Articles**: 2,251 (5.6% of topic corpus)

**Description**: Military policy and procurement, defense industry and arms exports, security policy and intelligence, crime prevention and law enforcement, disaster response.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Military Policy & Procurement | 1,080 | 48.0% | Swiss Military Policy (332), Swiss Military Procurement (115), Swiss Defense Policy (112) |
| Security Policy & Intelligence | 408 | 18.1% | Swiss Security Policy (73), Cybersecurity (45), Security Threats (44) |
| Crime Prevention & Law Enforcement | 278 | 12.4% | Swiss Crime Policy (22), Swiss Crime Prevention (20), Traffic Regulation (18) |
| Disaster Response & Humanitarian | 221 | 9.8% | Swiss Disaster Response (41), Disaster Relief (35), Swiss Gaza Policy (32) |
| Other Defense & Security Topics | 148 | 6.6% | Defense Policy (18), Security (16), Military Cooperation (8) |
| Defense Industry & Arms Exports | 87 | 3.9% | Swiss Defense Industry (34), Swiss Waffenexport Policy (21), Arms Export Policy (12) |

### Environment, Energy & Infrastructure
**Articles**: 1,586 (4.0% of topic corpus)

**Description**: Energy policy and nuclear, climate and environmental protection, transportation and infrastructure, urban planning and development.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Energy Policy & Nuclear | 529 | 33.4% | Swiss Energy Policy (263), Energy Policy (105), Swiss Nuclear Energy Policy (29) |
| Climate & Environmental Protection | 476 | 30.0% | Swiss Climate Policy (103), Swiss Environmental Policy (78), Climate Policy (62) |
| Transportation & Infrastructure | 452 | 28.5% | Swiss Transportation Infrastru (173), Swiss Transportation Policy (80), Transport Infrastructure (44) |
| Other Environment & Infrastructure Topics | 97 | 6.1% | Accessibility in Public Transp (7), Natural Disasters (7), Swiss Internet Infrastructure (6) |
| Urban Planning & Development | 15 | 0.9% | Swiss Urbanization (7), Swiss Urban Policy (5), Swiss Urban Violence Preventio (3) |

### Healthcare, Social Welfare & Education
**Articles**: 1,411 (3.5% of topic corpus)

**Description**: Healthcare policy and reform, education policy and reform, social benefits and family policy, end-of-life and disability policy.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Healthcare Policy & Reform | 759 | 53.8% | Swiss Healthcare Policy (212), Healthcare Reform (172), Healthcare Policy (77) |
| Education Policy & Reform | 369 | 26.2% | Swiss Education Policy (178), Education Reform (58), Swiss Education Reform (50) |
| Social Benefits & Family Policy | 145 | 10.3% | Swiss Family Policy (30), Pension Reform (28), SocialWelfare (15) |
| Other Health & Welfare Topics | 42 | 3.0% | Healthcare (5), Youth Education (5), Swiss Women's Pension Policy (4) |
| End-of-Life, Mental Health & Disability | 37 | 2.6% | Suizidhilfegesetz (16), Suizidhilfe (6), Euthanasia Controversy (5) |

### International Relations & Geopolitics
**Articles**: 1,283 (3.2% of topic corpus)

**Description**: Switzerland-Ukraine relations, Middle East and Africa policy, Switzerland-US relations, Switzerland-Asia relations, Swiss neutrality and foreign aid.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Switzerland-Ukraine & Eastern Europe | 329 | 25.6% | Swiss Ukraine Relations (195), Swiss Ukraine Diplomacy (88), Swiss Ukraine Aid (11) |
| Swiss Neutrality & Foreign Aid | 326 | 25.4% | Swiss Foreign Policy (103), Swiss International Relations (47), Swiss Foreign Aid Policy (21) |
| Middle East & Africa Policy | 210 | 16.4% | Swiss Hamas Policy (46), Swiss Israel Relations (20), Swiss Palestine Relations (19) |
| Switzerland-US Relations | 163 | 12.7% | US Swiss Relations (92), US-Swiss Relations (22), US Swiss Relations is not appl (18) |
| Switzerland-Asia Relations | 66 | 5.1% | Swiss China Relations (64), Uigur Women's Rights (1), Karabakh Conflict (1) |
| Other International Topics | 61 | 4.8% | Swiss Afghan Relations (6), International Relations (6), Swiss Gastronomy Awards (6) |

### Demographic Change & Population
**Articles**: 292 (0.7% of topic corpus)

**Description**: Aging society and population trends, population and immigration demographics, fertility and pronatalism. Provides demographic context cited in initiative arguments.

**Subgroups** (sorted by size):

| Subgroup | Articles | % of cat. | Top labels |
|----------|----------|-----------|------------|
| Aging Society & Population Trends | 252 | 86.3% | Demographic Change (87), Demografische Entwicklung (69), Demografischer Wandel (53) |
| Population & Immigration Demographics | 39 | 13.4% | Population Growth (12), Demography Policy (10), Demografie (7) |
| Fertility & Pronatalism | 1 | 0.3% | Pronatalismus (1) |

## Supporting Files

All files are in `results/`:

| File | Description |
|------|-------------|
| `German_politic_20260503_194531.json` | Main corpus — each record has `id`, `head`, `pubtime`, `medium_code`, `classification`, `summary`, `sentiment` |
| `subtopic_mapping_10cat.json` | Maps every raw LLM subtopic label → major category. Fields: `subtopic_to_category`, `category_counts_from_summary` |
| `subgroup_taxonomy.json` | 3-level taxonomy: category → subgroup → top raw labels with counts |
| `topic_statistics_report_v2.pdf` | Visual report with bar charts and subgroup tables |

## Record Schema

```json
{
  "id": 51465542,
  "source_file": "German_politic.tsv",
  "row_index": 63084,
  "pubtime": "2023-10-10 00:00:00+02",
  "medium_code": "LUZ",
  "language": "de",
  "head": "Article headline (German)",
  "classification": "Immigration & Asylum Policy",  // one of the 10 categories or NOT TOPIC
  "raw_response": "Swiss Asylum Policy",            // original granular LLM label
  "verification": "yes",                            // yes = relevant to initiative
  "sentiment": "negative",
  "summary": "English summary of article..."
}
```

## Mapping File Schema

```json
// subtopic_mapping_10cat.json
{
  "subtopic_to_category": {
    "Swiss Asylum Policy": "Immigration & Asylum Policy",
    "Swiss EU Relations": "Swiss-EU Relations & Bilateral Agreements",
    "NOT TOPIC": "NOT TOPIC",
    ...
  },
  "category_counts_from_summary": { "Immigration & Asylum Policy": 5751, ... }
}
```