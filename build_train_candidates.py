"""
build_train_candidates.py
=========================
Generates output/train_candidate_pairs.tsv for train S1 entities
using DuckDB multi-route progressive blocking.
"""

import os
import time
import duckdb

def run_train_blocking(sample_size=None):
    start_time = time.time()
    os.makedirs("output", exist_ok=True)
    os.makedirs("duckdb_spill", exist_ok=True)

    db_path = "duckdb_spill/blocking_train.duckdb"
    if os.path.exists(db_path):
        os.remove(db_path)

    con = duckdb.connect(db_path)
    
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA max_memory='6GB';")
    con.execute("PRAGMA preserve_insertion_order=false;")
    con.execute("PRAGMA temp_directory='duckdb_spill';")

    print("=" * 65)
    print("STEP 1: Indexing Train Datasets into DuckDB...")
    print("=" * 65)

    limit_clause = f"LIMIT {sample_size}" if sample_size else ""

    con.execute(f"""
        CREATE TABLE s1 AS 
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
            regexp_extract(business_address_normalized, '([0-9]{2,6})', 1) AS addr_num,
            row_number() OVER () AS row_idx
        FROM read_csv('data/processed/train_source1_processed.tsv', sep='\\t')
        {limit_clause};

        CREATE TABLE s23 AS 
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
            SELECT entity_id, business_name_normalized, business_address_normalized, country_clean FROM read_csv('data/processed/train_source2_processed.tsv', sep='\\t')
            UNION ALL
            SELECT entity_id, business_name_normalized, business_address_normalized, country_clean FROM read_csv('data/processed/train_source3_processed.tsv', sep='\\t')
        );

        CREATE TABLE final_capped_candidates (
            s1_id VARCHAR,
            target_id VARCHAR,
            PRIMARY KEY (s1_id, target_id)
        );
    """)

    countries = [r[0] for r in con.execute("SELECT DISTINCT country_clean FROM s1 WHERE country_clean IS NOT NULL;").fetchall()]
    print(f"Countries to process: {countries}")

    print("=" * 65)
    print("STEP 2: Executing Staged Progressive Blocking for Train...")
    print("=" * 65)

    for country in countries:
        t0 = time.time()
        total_country_s1 = con.execute(f"SELECT count(*) FROM s1 WHERE country_clean = '{country}';").fetchone()[0]
        print(f"\n---> Partition: [{country}] ({total_country_s1:,} S1 entities)")

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE cur_s23 AS 
            SELECT entity_id, business_name_normalized, clean_name, clean_addr, addr_num
            FROM s23 WHERE country_clean = '{country}';

            CREATE OR REPLACE TEMP TABLE ft1 AS 
            SELECT split_part(clean_name, ' ', 1) AS tok FROM cur_s23 GROUP BY tok HAVING count(*) > 800 OR length(tok) < 3;

            CREATE OR REPLACE TEMP TABLE ft2 AS 
            SELECT split_part(clean_name, ' ', 2) AS tok FROM cur_s23 GROUP BY tok HAVING count(*) > 600 OR length(tok) < 4;

            CREATE OR REPLACE TEMP TABLE ft3 AS 
            SELECT split_part(clean_name, ' ', 3) AS tok FROM cur_s23 GROUP BY tok HAVING count(*) > 600 OR length(tok) < 4;

            CREATE OR REPLACE TEMP TABLE fa AS 
            SELECT substring(clean_addr, 1, 9) AS ap FROM cur_s23 WHERE length(clean_addr) >= 9 GROUP BY ap HAVING count(*) > 400;
        """)

        batch_size = 100000
        for offset in range(0, total_country_s1, batch_size):
            b_start = time.time()
            print(f"   Batch {offset:,} to {min(offset + batch_size, total_country_s1):,}...")

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE batch_s1 AS 
                SELECT entity_id, business_name_normalized, clean_name, clean_addr, addr_num
                FROM s1 
                WHERE country_clean = '{country}'
                ORDER BY row_idx
                LIMIT {batch_size} OFFSET {offset};
            """)

            routes = [
                ("Route 1: Exact Name", """
                    SELECT DISTINCT s1.entity_id AS s1_id, s23.entity_id AS target_id 
                    FROM batch_s1 s1 JOIN cur_s23 s23 
                      ON s1.business_name_normalized = s23.business_name_normalized
                    WHERE length(s1.business_name_normalized) >= 3
                """),
                ("Route 2: Token 1", """
                    SELECT DISTINCT s1.entity_id AS s1_id, s23.entity_id AS target_id 
                    FROM batch_s1 s1 JOIN cur_s23 s23 
                      ON split_part(s1.clean_name, ' ', 1) = split_part(s23.clean_name, ' ', 1)
                    LEFT JOIN ft1 ON split_part(s1.clean_name, ' ', 1) = ft1.tok
                    WHERE ft1.tok IS NULL AND length(split_part(s1.clean_name, ' ', 1)) >= 3
                """),
                ("Route 3: Cross Token 1 <-> 2", """
                    SELECT DISTINCT s1.entity_id AS s1_id, s23.entity_id AS target_id 
                    FROM batch_s1 s1 JOIN cur_s23 s23 
                      ON split_part(s1.clean_name, ' ', 1) = split_part(s23.clean_name, ' ', 2)
                    LEFT JOIN ft2 ON split_part(s1.clean_name, ' ', 1) = ft2.tok
                    WHERE ft2.tok IS NULL AND length(split_part(s1.clean_name, ' ', 1)) >= 4
                """),
                ("Route 4: Cross Token 2 <-> 1", """
                    SELECT DISTINCT s1.entity_id AS s1_id, s23.entity_id AS target_id 
                    FROM batch_s1 s1 JOIN cur_s23 s23 
                      ON split_part(s1.clean_name, ' ', 2) = split_part(s23.clean_name, ' ', 1)
                    LEFT JOIN ft1 ON split_part(s1.clean_name, ' ', 2) = ft1.tok
                    WHERE ft1.tok IS NULL AND length(split_part(s1.clean_name, ' ', 2)) >= 4
                """),
                ("Route 5: Token 2", """
                    SELECT DISTINCT s1.entity_id AS s1_id, s23.entity_id AS target_id 
                    FROM batch_s1 s1 JOIN cur_s23 s23 
                      ON split_part(s1.clean_name, ' ', 2) = split_part(s23.clean_name, ' ', 2)
                    LEFT JOIN ft2 ON split_part(s1.clean_name, ' ', 2) = ft2.tok
                    WHERE ft2.tok IS NULL AND length(split_part(s1.clean_name, ' ', 2)) >= 4
                """),
                ("Route 6: Token 3", """
                    SELECT DISTINCT s1.entity_id AS s1_id, s23.entity_id AS target_id 
                    FROM batch_s1 s1 JOIN cur_s23 s23 
                      ON split_part(s1.clean_name, ' ', 3) = split_part(s23.clean_name, ' ', 3)
                    LEFT JOIN ft3 ON split_part(s1.clean_name, ' ', 3) = ft3.tok
                    WHERE ft3.tok IS NULL AND length(split_part(s1.clean_name, ' ', 3)) >= 4
                """),
                ("Route 7: Address Prefix (9 chars)", """
                    SELECT DISTINCT s1.entity_id AS s1_id, s23.entity_id AS target_id 
                    FROM batch_s1 s1 JOIN cur_s23 s23 
                      ON substring(s1.clean_addr, 1, 9) = substring(s23.clean_addr, 1, 9)
                    LEFT JOIN fa ON substring(s1.clean_addr, 1, 9) = fa.ap
                    WHERE fa.ap IS NULL AND length(s1.clean_addr) >= 9
                """),
                ("Route 8: Street Number + 3-char Name", """
                    SELECT DISTINCT s1.entity_id AS s1_id, s23.entity_id AS target_id 
                    FROM batch_s1 s1 JOIN cur_s23 s23 
                      ON s1.addr_num = s23.addr_num
                     AND substring(s1.clean_name, 1, 3) = substring(s23.clean_name, 1, 3)
                    WHERE s1.addr_num != '' AND length(s1.clean_name) >= 3
                """)
            ]

            for _, sql in routes:
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE step_pairs AS
                    SELECT s1_id, target_id FROM ({sql})
                    WHERE s1_id IN (
                        SELECT entity_id FROM batch_s1 
                        EXCEPT 
                        SELECT s1_id FROM final_capped_candidates GROUP BY s1_id HAVING count(*) >= 15
                    );

                    INSERT OR IGNORE INTO final_capped_candidates
                    WITH cur_counts AS (
                        SELECT s1_id, count(*) AS cur_cnt FROM final_capped_candidates GROUP BY s1_id
                    ),
                    ranked AS (
                        SELECT p.s1_id, p.target_id,
                               ROW_NUMBER() OVER (PARTITION BY p.s1_id ORDER BY p.target_id) AS rn,
                               COALESCE(c.cur_cnt, 0) AS cnt
                        FROM step_pairs p
                        LEFT JOIN cur_counts c ON p.s1_id = c.s1_id
                    )
                    SELECT s1_id, target_id FROM ranked WHERE (cnt + rn) <= 15;

                    DROP TABLE step_pairs;
                """)

            print(f"   Batch finished in {time.time() - b_start:.1f}s")

        con.execute("""
            DROP TABLE cur_s23;
            DROP TABLE ft1;
            DROP TABLE ft2;
            DROP TABLE ft3;
            DROP TABLE fa;
        """)

        cnt = con.execute("SELECT count(*) FROM final_capped_candidates;").fetchone()[0]
        print(f"  Finished [{country}] in {time.time() - t0:.2f}s. Total candidates: {cnt:,}")

    print("=" * 65)
    print("STEP 3: Aggregating & Exporting output/train_candidate_pairs.tsv...")
    print("=" * 65)

    con.execute("""
        COPY (
            SELECT 
                s1.entity_id AS source1_entity_id,
                COALESCE(string_agg(c.target_id, ','), '') AS candidate_entity_ids
            FROM s1
            LEFT JOIN final_capped_candidates c ON s1.entity_id = c.s1_id
            GROUP BY s1.entity_id
            ORDER BY s1.entity_id
        ) TO 'output/train_candidate_pairs.tsv' (HEADER TRUE, DELIMITER '\\t');
    """)

    elapsed = time.time() - start_time
    print(f"\nSUCCESS! output/train_candidate_pairs.tsv generated in {elapsed:.2f}s.")

    con.close()

if __name__ == "__main__":
    import sys
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    print(f"Running train blocking for sample_size={sample_size}...")
    run_train_blocking(sample_size=sample_size)
