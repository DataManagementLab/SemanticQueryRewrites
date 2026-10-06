"""Per-dataset domain knowledge for rendering generation prompts + rule examples.

Each entry feeds the shared prompt scaffold in ``render_prompts.py`` (which mirrors the
hand-written ``prompts/basketball/generation/prompt_wk01.txt`` and
``prompts/imdb_job/generation/prompt12.txt``). The variable parts are:

    display          - noun phrase describing the database (used after "over ...")
    adjective        - "<adjective>-specific shortcuts" (cf. "basketball-specific")
    knowledge        - "Leverage <knowledge> to justify simplifications"
    data_ref         - "realistic for <data_ref>" / "safe for <data_ref>"
    bullets          - domain world-knowledge shortcuts (the substantive content)
    example_rule     - a format-correct rule over real columns of this dataset

The domains were inferred from the dataset name + table/column names on the lab mount.
For synthetic (tpc_h, ssb) and foreign-language / cryptic schemas (accidents [Slovenian],
geneea/fhnk/seznam [Czech], carcinogenesis [molecular]) the bullets emphasise cross-table
join-implied filters and value-range correlations rather than famous entities.

``FULL_NAMES`` maps each project-local short name to its ``_scaledN`` folder on the mount.
"""

from __future__ import annotations

FULL_NAMES: dict[str, str] = {
    "accidents": "accidents_scaled1",
    "airline": "airline_scaled1",
    "baseball": "baseball_scaled10",
    "carcinogenesis": "carcinogenesis_scaled674",
    "consumer": "consumer_scaled6",
    "credit": "credit_scaled5",
    "employee": "employee_scaled3",
    "fhnk": "fhnk_scaled2",
    "financial": "financial_scaled4",
    "geneea": "geneea_scaled23",
    "genome": "genome_scaled6",
    "hepatitis": "hepatitis_scaled2000",
    "movielens": "movielens_scaled8",
    "seznam": "seznam_scaled2",
    "ssb": "ssb_scaled1",
    "tournament": "tournament_scaled50",
    "tpc_h": "tpc_h_scaled1",
    "walmart": "walmart_scaled1",
}


def _rule(short: str, requires: dict, implies: list[dict]) -> dict:
    return {"id": f"{short.upper()}_R1", "requires": requires, "implies": implies}


def _and(*conditions: dict) -> dict:
    return {"op": "AND", "conditions": list(conditions)}


SPECS: dict[str, dict] = {
    "accidents": {
        "display": "a Slovenian road-traffic accident database (nesreca = accidents, "
                   "oseba = persons involved, upravna_enota = administrative units)",
        "adjective": "traffic-accident",
        "knowledge": "real-world knowledge about how road-traffic accidents are recorded",
        "data_ref": "this Slovenian accidents dataset",
        "bullets": [
            "Administrative-unit carry: nesreca and oseba both reference upravna_enota; a filter pinning one side's administrative unit usually holds on the joined side too.",
            "Age sanity: when a person is recorded as a driver, oseba.starost falls in a realistic adult range you can bound.",
            "Driving-experience bounds: oseba.vozniski_staz_LL (years of driving licence) is bounded by the person's age; a large experience filter implies an older driver.",
            "Severity correlation: severe values in oseba.poskodba (injury) tend to co-occur with particular accident circumstances on nesreca — narrow joined attributes accordingly.",
        ],
        "example_rule": _rule("ACCIDENTS",
            _and({"column": "oseba.vozniski_staz_LL", "op": ">=", "value": 30}),
            [{"column": "oseba.starost", "op": ">=", "value": 45}]),
    },
    "airline": {
        "display": "a U.S. airline on-time performance database (On_Time_On_Time_Performance_2016_1 "
                   "flight records plus L_* lookup tables for carriers, airports, cancellation reasons, etc.)",
        "adjective": "airline",
        "knowledge": "real-world aviation knowledge",
        "data_ref": "this airline on-time dataset",
        "bullets": [
            "Single-year scope: this is the 2016 performance extract, so Year/Quarter/Month are tightly bounded — add a Year filter when the query implies a period.",
            "Cancellation logic: a cancelled flight (Cancelled = 1) was not diverted and has null/zero delay columns; ArrDel15 = 1 implies ArrDelayMinutes >= 15.",
            "Geography carry: a filter on OriginState / OriginCityName implies the matching OriginStateFips / OriginWac; the same holds on the Dest side.",
            "Diverted flights: DivAirportLandings > 0 implies Diverted = 1 and that the Div1* columns are populated.",
            "Distance groups: a DistanceGroup filter implies a Distance range and vice versa.",
        ],
        "example_rule": _rule("AIRLINE",
            _and({"column": "On_Time_On_Time_Performance_2016_1.Cancelled", "op": "=", "value": 1}),
            [{"column": "On_Time_On_Time_Performance_2016_1.Diverted", "op": "=", "value": 0}]),
    },
    "baseball": {
        "display": "a historical baseball statistics database (Lahman: players, teams, batting, "
                   "pitching, fielding, all-star, hall-of-fame, awards, postseason)",
        "adjective": "baseball",
        "knowledge": "real-world baseball knowledge",
        "data_ref": "this baseball dataset",
        "bullets": [
            "League carry: a lgID filter (AL/NL) on one table usually holds on the joined batting/pitching/teams rows for the same player-year.",
            "Era / year sanity: leagues, awards and events imply a yearID range you can bound (the National League dates from 1876; many awards start far later).",
            "Postseason: joining battingpost/pitchingpost/seriespost or filtering round implies the team reached the playoffs that year — narrow teams accordingly.",
            "Award implications: an all-star or hall-of-fame filter implies non-trivial career stats and a plausible debut/finalGame range.",
            "Well-known franchises/players: use known franchise eras and player careers to bound yearID/teamID.",
        ],
        "example_rule": _rule("BASEBALL",
            _and({"column": "teams.lgID", "op": "=", "value": "NL"}),
            [{"column": "teams.yearID", "op": ">=", "value": 1876}]),
    },
    "carcinogenesis": {
        "display": "a molecular carcinogenesis database (atom = atoms, sbond_* = bonds, "
                   "and drugs with their cancer classification in canc)",
        "adjective": "molecular",
        "knowledge": "real-world chemistry knowledge about molecular structures",
        "data_ref": "this carcinogenesis dataset",
        "bullets": [
            "Drug carry: atom, sbond_* and canc all reference a drug id; a filter pinning one molecule propagates to all of its atoms and bonds.",
            "Bond-table semantics: sbond_1/2/3/7 encode bond multiplicity; selecting one restricts the bond order and the atom pairs it connects.",
            "Atom-type / charge correlation: an atomtype filter (a specific element) implies a realistic charge range on atom.charge.",
            "Connectivity: a bond row links two atomids of the same drug — a filter on one atom's properties can be carried to the bonded atom via the join.",
        ],
        "example_rule": _rule("CARCINOGENESIS",
            _and({"column": "atom.atomtype", "op": "=", "value": "c"}),
            [{"column": "atom.charge", "op": ">=", "value": -1}]),
    },
    "consumer": {
        "display": "a U.S. consumer-expenditure survey database (HOUSEHOLDS, HOUSEHOLD_MEMBERS, "
                   "and EXPENDITURES by product/year/month)",
        "adjective": "consumer-spending",
        "knowledge": "real-world knowledge about household spending surveys",
        "data_ref": "this consumer-expenditure dataset",
        "bullets": [
            "Household carry: EXPENDITURES, HOUSEHOLDS and HOUSEHOLD_MEMBERS share HOUSEHOLD_ID and YEAR; a year/household filter on one side holds on the joined side.",
            "Income-rank consistency: a HOUSEHOLDS.INCOME_RANK band implies the INCOME_RANK_1..5 percentile columns fall in a corresponding range.",
            "Gift vs cost: an EXPENDITURES.GIFT flag implies a non-null, positive COST.",
            "Member demographics: a WORK_STATUS or MARITAL filter on HOUSEHOLD_MEMBERS implies a plausible AGE range.",
        ],
        "example_rule": _rule("CONSUMER",
            _and({"column": "EXPENDITURES.GIFT", "op": "=", "value": 1}),
            [{"column": "EXPENDITURES.COST", "op": ">", "value": 0}]),
    },
    "credit": {
        "display": "a credit-card membership database (members, corporations, providers, charges, "
                   "payments, statements, regions, categories)",
        "adjective": "credit-card",
        "knowledge": "real-world knowledge about credit-card membership and billing",
        "data_ref": "this credit-card dataset",
        "bullets": [
            "Region carry: member, provider, corporation and region all carry region_no; a region filter on one side usually holds on the joined side.",
            "Corporate membership: a corp_no filter on member implies the member shares the corporation's region and mail-code patterns.",
            "Billing logic: a charge joined to a statement implies the charge_dt falls within the statement period; payments reference the same member_no as their statement.",
            "Balance sanity: a curr_balance/prev_balance filter implies plausible charge_amt and payment_amt ranges.",
        ],
        "example_rule": _rule("CREDIT",
            _and({"column": "provider.region_no", "op": "=", "value": 5}),
            [{"column": "provider.country", "op": "=", "value": "USA"}]),
    },
    "employee": {
        "display": "an HR / employees database (departments, employees, dept_emp, dept_manager, "
                   "salaries, titles — each history table carrying from_date/to_date)",
        "adjective": "HR-employee",
        "knowledge": "real-world knowledge about employment records",
        "data_ref": "this employees dataset",
        "bullets": [
            "Date-range scope: this sample spans hire_dates and from_date/to_date in a known historical window — bound dates when the query implies a period.",
            "Current vs historical: to_date = '9999-01-01' marks the current row in dept_emp/salaries/titles; a 'current' filter implies that sentinel.",
            "Tenure logic: a from_date filter on salaries/titles implies hire_date <= from_date for the same emp_no.",
            "Age/birth sanity: birth_date and hire_date imply a plausible adult age at hire.",
        ],
        "example_rule": _rule("EMPLOYEE",
            _and({"column": "titles.to_date", "op": "=", "value": "9999-01-01"}),
            [{"column": "titles.from_date", "op": "<=", "value": "2003-01-01"}]),
    },
    "fhnk": {
        "display": "a Czech hospital records database (pripady = patient cases, vykony = medical "
                   "procedures, zup = special-material procedures)",
        "adjective": "hospital",
        "knowledge": "real-world knowledge about hospital case records",
        "data_ref": "this hospital (FHNK) dataset",
        "bullets": [
            "Case carry: pripady, vykony and zup share Identifikace_pripadu; a filter on one side's case attributes holds on the joined procedures.",
            "Admission/discharge logic: Delka_hospitalizace (length of stay) is the gap between Datum_prijeti and Datum_propusteni — a long-stay filter implies a date span and a likely DRG group.",
            "Procedure dates: vykony/zup Datum_provedeni_vykonu falls between the case's admission and discharge dates.",
            "Demographic bounds: a Vekovy_Interval_Pacienta (age interval) filter narrows the plausible Zakladni_diagnoza set.",
        ],
        "example_rule": _rule("FHNK",
            _and({"column": "vykony.Pocet", "op": ">", "value": 0}),
            [{"column": "vykony.Body", "op": ">=", "value": 0}]),
    },
    "financial": {
        "display": "a Czech bank database (accounts, clients, dispositions, cards, loans, orders, "
                   "transactions, districts)",
        "adjective": "banking",
        "knowledge": "real-world banking knowledge",
        "data_ref": "this financial (Czech bank) dataset",
        "bullets": [
            "District carry: account and client both reference district_id; a district filter on one side usually holds on the joined side.",
            "Disposition logic: disp.type = 'OWNER' is the account owner; cards (card.disp_id) only exist for dispositions, so a card join implies an existing disposition.",
            "Loan vs balance: a loan filter implies the account had transaction history; loan.status encodes the repayment state.",
            "Transaction sanity: trans.type/operation imply the sign and plausible ranges of trans.amount and trans.balance.",
            "Date window: this bank dataset covers a known 1990s window — bound dates when the query implies a period.",
        ],
        "example_rule": _rule("FINANCIAL",
            _and({"column": "loan.status", "op": "IN", "value": ["A", "C"]}),
            [{"column": "loan.amount", "op": ">", "value": 0}]),
    },
    "geneea": {
        "display": "a Czech parliament (Poslanecká sněmovna) voting database (osoby = persons, "
                   "poslanec = MPs, hl_hlasovani = votes, organy = bodies, funkce = functions)",
        "adjective": "parliamentary",
        "knowledge": "real-world knowledge about parliamentary voting records",
        "data_ref": "this Czech parliament dataset",
        "bullets": [
            "Person carry: osoby, poslanec, hl_poslanec and zarazeni reference the same id_osoba/id_poslanec; a person filter propagates across the joins.",
            "Body/term scope: an id_organ filter implies a term (od_organ..do_organ) date range you can bound.",
            "Vote outcome: hl_hlasovani.vysledek (passed/failed) correlates with the pro/proti/zdrzel tallies and the quorum (kvorum).",
            "Membership dates: zarazeni od_o..do_o (function periods) fall within the corresponding organ's existence.",
        ],
        "example_rule": _rule("GENEEA",
            _and({"column": "hl_hlasovani.pro", "op": ">=", "value": 101}),
            [{"column": "hl_hlasovani.vysledek", "op": "=", "value": "A"}]),
    },
    "genome": {
        "display": "a Visual Genome image scene-graph database (images and their objects (IMG_OBJ), "
                   "object attributes (IMG_OBJ_ATT), relations (IMG_REL), and class lookups)",
        "adjective": "scene-graph",
        "knowledge": "real-world knowledge about image scene graphs",
        "data_ref": "this Visual Genome dataset",
        "bullets": [
            "Image carry: IMG_OBJ, IMG_OBJ_ATT and IMG_REL share IMG_ID; an image filter on one side holds on the joined side.",
            "Object-sample carry: attributes and relations reference OBJ_SAMPLE_ID within an image — pin the object on both sides of the join.",
            "Bounding-box sanity: IMG_OBJ X/Y/W/H are non-negative and bounded by the image dimensions; a width/height filter implies a coordinate range.",
            "Class consistency: an OBJ_CLASS_ID / ATT_CLASS_ID / PRED_CLASS_ID filter maps to exactly one class label in the lookup tables.",
        ],
        "example_rule": _rule("GENOME",
            _and({"column": "IMG_OBJ.W", "op": ">", "value": 0}),
            [{"column": "IMG_OBJ.H", "op": ">", "value": 0}]),
    },
    "hepatitis": {
        "display": "a hepatitis study database (dispat = patients, indis = lab indicators, "
                   "Bio = biopsy, inf = infection duration, with rel11/rel12/rel13 link tables)",
        "adjective": "hepatitis-study",
        "knowledge": "real-world medical knowledge about hepatitis",
        "data_ref": "this hepatitis dataset",
        "bullets": [
            "Patient carry: dispat (m_id) links to Bio, indis and inf through rel11/rel12/rel13; a patient filter propagates across the joins.",
            "Lab-value correlation: liver-enzyme indicators in indis (got, gpt, ...) co-vary — a high-value filter on one enzyme implies plausible ranges on related ones.",
            "Fibrosis/activity staging: Bio.fibros and Bio.activity stages correlate; an advanced stage on one implies a range on the other.",
            "Demographic bounds: a dispat.sex/Type filter narrows plausible dispat.age ranges.",
        ],
        "example_rule": _rule("HEPATITIS",
            _and({"column": "indis.gpt", "op": ">", "value": 100}),
            [{"column": "indis.got", "op": ">", "value": 0}]),
    },
    "movielens": {
        "display": "a MovieLens movie-ratings database (movies, actors, directors, users, "
                   "movie–actor and movie–director links, and u2base ratings)",
        "adjective": "movie-ratings",
        "knowledge": "real-world movie knowledge",
        "data_ref": "this MovieLens dataset",
        "bullets": [
            "Rating scale: u2base.rating is on a fixed bounded scale; a 'high rating' filter implies a bounded numeric range.",
            "Language/country carry: movies.isEnglish and movies.country are consistent — an isEnglish filter implies a country set and vice versa.",
            "Year sanity: a genre or director-era filter implies a plausible movies.year range.",
            "Quality correlation: actors.a_quality / directors.d_quality bands correlate with directors.avg_revenue and rating distributions.",
            "Demographics: a users.occupation / u_gender filter implies a plausible users.age range.",
        ],
        "example_rule": _rule("MOVIELENS",
            _and({"column": "u2base.rating", "op": ">=", "value": 4}),
            [{"column": "u2base.rating", "op": "<=", "value": 5}]),
    },
    "seznam": {
        "display": "a Seznam.cz online-advertising wallet database (clients, dobito = wallet top-ups, "
                   "probehnuto = clicked-through spend, probehnuto_mimo_penezenku = off-wallet advertising)",
        "adjective": "ad-wallet",
        "knowledge": "real-world knowledge about prepaid online-advertising wallets",
        "data_ref": "this Seznam advertising dataset",
        "bullets": [
            "Client carry: client, dobito and probehnuto share client_id; a client/region (kraj) or sector (obor) filter propagates across the joins.",
            "Spend vs top-up: probehnuto.kc_proklikano (spent) is bounded over time by the client's dobito.kc_dobito (topped-up).",
            "Service consistency: a sluzba (service) filter on one transaction table narrows the services seen on the joined table for the same client.",
            "Period carry: the month_year_datum_transakce period on top-ups and click-through tends to align for an active client.",
        ],
        "example_rule": _rule("SEZNAM",
            _and({"column": "dobito.kc_dobito", "op": ">=", "value": 1000}),
            [{"column": "dobito.kc_dobito", "op": ">", "value": 0}]),
    },
    "ssb": {
        "display": "a Star Schema Benchmark (SSB) sales database (lineorder fact joined to "
                   "customer, supplier, part and dim_date dimensions)",
        "adjective": "sales-OLAP",
        "knowledge": "knowledge about the SSB star schema and its synthetic data distributions",
        "data_ref": "this SSB dataset",
        "bullets": [
            "Dimension carry: lineorder joins to dim_date on lo_orderdate = d_datekey; a d_year/d_yearmonth filter implies a bounded lo_orderdate range.",
            "Geography carry: a customer c_region/c_nation filter implies the matching c_city prefix; the same holds for supplier s_region/s_nation/s_city.",
            "Part hierarchy: p_category determines p_mfgr and constrains p_brand1; a brand filter implies its category.",
            "Revenue identity: lo_revenue relates to lo_extendedprice and lo_discount — a discount/quantity filter bounds revenue.",
        ],
        "example_rule": _rule("SSB",
            _and({"column": "lineorder.lo_quantity", "op": ">", "value": 0}),
            [{"column": "lineorder.lo_extendedprice", "op": ">", "value": 0}]),
    },
    "tournament": {
        "display": "an NCAA basketball tournament database (teams, regular-season and tournament "
                   "results, seeds, slots, seasons, and a prediction target table)",
        "adjective": "tournament",
        "knowledge": "real-world knowledge about NCAA basketball tournaments",
        "data_ref": "this NCAA tournament dataset",
        "bullets": [
            "Season carry: results, seeds and slots share season; a season filter on one side holds on the joined side.",
            "Tournament implication: a row in tourney_* (or a tourney_seeds entry) implies the team qualified for that season's tournament.",
            "Score sanity: the winner score wscore exceeds the loser score lscore; a margin/score filter implies bounded point totals.",
            "Seed semantics: a tourney_seeds.seed band correlates with stronger regular-season results (more wins, larger margins).",
        ],
        "example_rule": _rule("TOURNAMENT",
            _and({"column": "tourney_detailed_results.wscore", "op": ">", "value": 0}),
            [{"column": "tourney_detailed_results.lscore", "op": ">=", "value": 0}]),
    },
    "tpc_h": {
        "display": "a TPC-H decision-support database (lineitem, orders, customer, supplier, part, "
                   "partsupp, nation, region)",
        "adjective": "TPC-H",
        "knowledge": "knowledge about the TPC-H schema and its generated data distributions",
        "data_ref": "this TPC-H dataset",
        "bullets": [
            "Region/nation carry: customer and supplier join to nation→region; a region filter implies the matching nation set.",
            "Order-line logic: lineitem.l_shipdate <= l_receiptdate and both relate to the parent order's o_orderdate — a date filter on one bounds the others.",
            "Returned items: l_returnflag = 'R' implies a finalised line status (l_linestatus = 'F').",
            "Price identity: l_extendedprice relates to l_quantity and the part's p_retailprice; a quantity filter bounds extended price.",
            "Segment carry: a customer c_mktsegment filter narrows plausible order priorities.",
        ],
        "example_rule": _rule("TPC_H",
            _and({"column": "lineitem.l_returnflag", "op": "=", "value": "R"}),
            [{"column": "lineitem.l_linestatus", "op": "=", "value": "F"}]),
    },
    "walmart": {
        "display": "a Walmart store-sales database (train = unit sales by date/store/item, "
                   "key = store→weather-station mapping, station)",
        "adjective": "retail-sales",
        "knowledge": "real-world knowledge about retail store sales",
        "data_ref": "this Walmart sales dataset",
        "bullets": [
            "Store carry: train and key share store_nbr, and key maps store_nbr→station_nbr; a store filter propagates to its weather station.",
            "Date window: train.date covers a known multi-year window — bound dates when the query implies a period.",
            "Sales sanity: train.units is non-negative; a positive-sales filter excludes zero/closed-day rows.",
            "Item scope: an item_nbr filter narrows the stores that ever stocked that item.",
        ],
        "example_rule": _rule("WALMART",
            _and({"column": "train.units", "op": ">", "value": 100}),
            [{"column": "train.units", "op": ">", "value": 0}]),
    },
}


def get_spec(short: str) -> dict:
    if short not in SPECS:
        raise KeyError(f"no domain spec for {short!r}; known: {sorted(SPECS)}")
    return SPECS[short]
