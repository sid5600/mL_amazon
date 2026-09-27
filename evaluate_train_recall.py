import os
import duckdb
import time

print("=" * 65)
print("FINAL RECALL SURGE: Cleaned Punctuation + Token Overlap (50k GT)")
print("=" * 65)
start_time = time.time()

os.makedirs("duckdb_spill", exist_ok=True)
db_file = "duckdb_spill/recall_bench.duckdb"
if os.path.exists(db_file):
    os.remove(db_file)

con = duckdb.connect(db_file)
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA max_memory='5GB';")
con.execute("PRAGMA temp_directory='duckdb_spill';")

con.execute("""
    CREATE TABLE gt_sample AS 
    SELECT source1_entity_id AS s1_id, 
           trim(unnest(string_split(matched_entity_ids, ','))) AS true_target_id
    FROM read_csv('train_ground_truth.tsv', sep='\\t')
    WHERE matched_entity_ids != '' AND matched_entity_ids IS NOT NULL
    LIMIT 50000;

    CREATE TABLE s1_base AS
    SELECT 
        entity_id,
        business_name_normalized,
        business_address_normalized,
        country_clean,
        -- Strip leading dots, symbols, honorifics, and entity designators
        regexp_replace(
            regexp_replace(business_name_normalized, '^[.\\-\\s#/]+', ''),
            '^(the|a|an|hotel|shree|sri|shri|dr|m/s|ms|pvt|private|ltd|limited|llc|inc|corp|co)\\s+', 
            ''
        ) AS clean_name,
        regexp_replace(
            regexp_replace(business_address_normalized, '^[.\\-\\s#/]+', ''),
            '^(h\\.?number|flat number|door number|door no|plot no|flat no|house no|old number|new number|number|no\\.?)\\s*', 
            ''
        ) AS clean_addr,
        regexp_extract(business_address_normalized, '([0-9]{2,6})', 1) AS addr_num
    FROM read_csv('train_source1_processed.tsv', sep='\\t') s1
    SEMI JOIN gt_sample gt ON s1.entity_id = gt.s1_id;

    CREATE TABLE s23_base AS
    SELECT 
        entity_id,
        business_name_normalized,
        business_address_normalized,
        country_clean,
        regexp_replace(
            regexp_replace(business_name_normalized, '^[.\\-\\s#/]+', ''),
            '^(the|a|an|hotel|shree|sri|shri|dr|m/s|ms|pvt|private|ltd|limited|llc|inc|corp|co)\\s+', 
            ''
        ) AS clean_name,
        regexp_replace(
            regexp_replace(business_address_normalized, '^[.\\-\\s#/]+', ''),
            '^(h\\.?number|flat number|door number|door no|plot no|flat no|house no|old number|new number|number|no\\.?)\\s*', 
            ''
        ) AS clean_addr,
        regexp_extract(business_address_normalized, '([0-9]{2,6})', 1) AS addr_num
    FROM (
        SELECT entity_id, business_name_normalized, business_address_normalized, country_clean FROM read_csv('train_source2_processed.tsv', sep='\\t')
        UNION ALL
        SELECT entity_id, business_name_normalized, business_address_normalized, country_clean FROM read_csv('train_source3_processed.tsv', sep='\\t')
    );

    CREATE TABLE captured_pairs (s1_id VARCHAR, target_id VARCHAR);
""")

countries = [r[0] for r in con.execute("SELECT DISTINCT country_clean FROM s1_base WHERE country_clean IS NOT NULL;").fetchall()]

for c in countries:
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cur_s1 AS SELECT * FROM s1_base WHERE country_clean = '{c}';
        CREATE OR REPLACE TEMP TABLE cur_s23 AS SELECT * FROM s23_base WHERE country_clean = '{c}';

        CREATE OR REPLACE TEMP TABLE ft1 AS 
        SELECT split_part(clean_name, ' ', 1) AS tok FROM cur_s23 GROUP BY tok HAVING count(*) > 1500 OR length(tok) < 3;

        CREATE OR REPLACE TEMP TABLE ft2 AS 
        SELECT split_part(clean_name, ' ', 2) AS tok FROM cur_s23 GROUP BY tok HAVING count(*) > 1500 OR length(tok) < 4;

        CREATE OR REPLACE TEMP TABLE ft3 AS 
        SELECT split_part(clean_name, ' ', 3) AS tok FROM cur_s23 GROUP BY tok HAVING count(*) > 1500 OR length(tok) < 4;

        CREATE OR REPLACE TEMP TABLE fa AS 
        SELECT substring(clean_addr, 1, 9) AS ap FROM cur_s23 WHERE length(clean_addr) >= 9 GROUP BY ap HAVING count(*) > 500;

        INSERT INTO captured_pairs
        -- Route 1: Exact Name
        SELECT s1.entity_id, s23.entity_id FROM cur_s1 s1 JOIN cur_s23 s23 
          ON s1.business_name_normalized = s23.business_name_normalized WHERE length(s1.business_name_normalized) >= 3
        UNION
        -- Route 2: First Distinctive Token
        SELECT s1.entity_id, s23.entity_id FROM cur_s1 s1 JOIN cur_s23 s23 
          ON split_part(s1.clean_name, ' ', 1) = split_part(s23.clean_name, ' ', 1)
          LEFT JOIN ft1 ON split_part(s1.clean_name, ' ', 1) = ft1.tok WHERE ft1.tok IS NULL AND length(split_part(s1.clean_name, ' ', 1)) >= 3
        UNION
        -- Route 3: Cross Token 1 <-> Token 2 (Catches inverted positions like 'sarthi' vs 'shri sarthi')
        SELECT s1.entity_id, s23.entity_id FROM cur_s1 s1 JOIN cur_s23 s23 
          ON split_part(s1.clean_name, ' ', 1) = split_part(s23.clean_name, ' ', 2)
          LEFT JOIN ft2 ON split_part(s1.clean_name, ' ', 1) = ft2.tok WHERE ft2.tok IS NULL AND length(split_part(s1.clean_name, ' ', 1)) >= 4
        UNION
        -- Route 4: Cross Token 2 <-> Token 1
        SELECT s1.entity_id, s23.entity_id FROM cur_s1 s1 JOIN cur_s23 s23 
          ON split_part(s1.clean_name, ' ', 2) = split_part(s23.clean_name, ' ', 1)
          LEFT JOIN ft1 ON split_part(s1.clean_name, ' ', 2) = ft1.tok WHERE ft1.tok IS NULL AND length(split_part(s1.clean_name, ' ', 2)) >= 4
        UNION
        -- Route 5: Token 2 Matching
        SELECT s1.entity_id, s23.entity_id FROM cur_s1 s1 JOIN cur_s23 s23 
          ON split_part(s1.clean_name, ' ', 2) = split_part(s23.clean_name, ' ', 2)
          LEFT JOIN ft2 ON split_part(s1.clean_name, ' ', 2) = ft2.tok WHERE ft2.tok IS NULL AND length(split_part(s1.clean_name, ' ', 2)) >= 4
        UNION
        -- Route 6: Token 3 Matching
        SELECT s1.entity_id, s23.entity_id FROM cur_s1 s1 JOIN cur_s23 s23 
          ON split_part(s1.clean_name, ' ', 3) = split_part(s23.clean_name, ' ', 3)
          LEFT JOIN ft3 ON split_part(s1.clean_name, ' ', 3) = ft3.tok WHERE ft3.tok IS NULL AND length(split_part(s1.clean_name, ' ', 3)) >= 4
        UNION
        -- Route 7: Clean Address Prefix (9 chars, symbol/noise stripped)
        SELECT s1.entity_id, s23.entity_id FROM cur_s1 s1 JOIN cur_s23 s23 
          ON substring(s1.clean_addr, 1, 9) = substring(s23.clean_addr, 1, 9)
          LEFT JOIN fa ON substring(s1.clean_addr, 1, 9) = fa.ap WHERE fa.ap IS NULL AND length(s1.clean_addr) >= 9
        UNION
        -- Route 8: Street Number + 3-char Name Prefix
        SELECT s1.entity_id, s23.entity_id FROM cur_s1 s1 JOIN cur_s23 s23 
          ON s1.addr_num = s23.addr_num AND substring(s1.clean_name, 1, 3) = substring(s23.clean_name, 1, 3)
          WHERE s1.addr_num != '' AND length(s1.clean_name) >= 3
        UNION
        -- Route 9: 4-char Name Prefix
        SELECT s1.entity_id, s23.entity_id FROM cur_s1 s1 JOIN cur_s23 s23 
          ON substring(s1.clean_name, 1, 4) = substring(s23.clean_name, 1, 4) WHERE length(s1.clean_name) >= 4;
    """)

res = con.execute("""
    SELECT 
        count(c.target_id) AS captured_true_pairs,
        count(gt.true_target_id) AS total_true_pairs,
        round(count(c.target_id) * 100.0 / count(gt.true_target_id), 2) AS candidate_recall_pct
    FROM gt_sample gt
    LEFT JOIN (SELECT DISTINCT s1_id, target_id FROM captured_pairs) c
      ON gt.s1_id = c.s1_id AND gt.true_target_id = c.target_id;
""").fetchone()

elapsed = time.time() - start_time
print(f"\nElapsed Time: {elapsed:.2f}s")
print(f"Captured True Pairs: {res[0]:,} / {res[1]:,}")
print(f"Final Candidate Recall Ceiling: {res[2]}%")

con.close()