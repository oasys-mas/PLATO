-- ============================================================================
-- WILDFIRE MARL — COMPLETE INSIGHT QUERIES (Postgres)
-- Usage:  \set exp_id 42
--         \i wildfire_all.sql
-- ============================================================================


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q0  SCOREBOARD — best checkpoint per model × grid × native/zeroshot    │
-- └──────────────────────────────────────────────────────────────────────────┘

WITH ranked AS (
    SELECT
        e.id                                                        AS exp_id,
        e.model_family,
        e.variation1 || '/' || COALESCE(e.variation2,'—')           AS variant,
        e.grid_size,
        CASE WHEN e.zero_shot = 1 THEN 'zs' ELSE 'native' END     AS mode,
        e.zero_shot_mode,
        er.avg_reward,
        er.avg_putouts,
        er.avg_burnouts,
        ROUND(er.avg_noop::numeric, 1)                              AS avg_noop,
        ROUND(er.avg_length::numeric, 1)                            AS avg_len,
        ROUND(er.avg_putouts_per_fight::numeric, 3)                 AS efficiency,
        e.checkpoint_path,
        ROW_NUMBER() OVER (
            PARTITION BY e.model_family, e.grid_size, e.zero_shot
            ORDER BY er.avg_reward DESC
        ) AS rk
    FROM experiments e
    JOIN eval_results er ON er.experiment_id = e.id
    WHERE e.domain = 'wildfire'
      AND e.run_type = 'execution'
)
SELECT exp_id, model_family, variant, grid_size, mode, zero_shot_mode,
       avg_reward, avg_putouts, avg_burnouts, avg_noop, avg_len, efficiency,
       checkpoint_path
FROM ranked
WHERE rk = 1
ORDER BY grid_size, mode, avg_reward DESC;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q1  AGENT REPORT CARD                                                  │
-- │     noop bucketed by REACHABLE fires (Manhattan ≤ 1, suppressant > 0)  │
-- └──────────────────────────────────────────────────────────────────────────┘
-- noop_0_fightable = no adjacent lit fire OR dry         → justified
-- noop_1_fightable = 1 adjacent lit fire, had suppressant → suspect
-- noop_2plus       = 2+ adjacent lit fires, had suppressant → bad policy
-- noop_dry         = suppressant was 0, forced noop regardless

WITH scope AS (
    SELECT ws.id AS step_id, ws.episode_id
    FROM wildfire_steps ws
    JOIN wildfire_episodes we  ON we.id = ws.episode_id
    JOIN eval_runs evr         ON evr.id = we.eval_run_id
    JOIN eval_results er       ON er.id = evr.eval_id
    WHERE er.experiment_id = :exp_id
),
fightable AS (
    SELECT
        wsa.step_id,
        wsa.agent_id,
        wsa.action_type,
        wsa.agent_reward,
        wsa.suppressant,
        wsa.target_fire_row,
        wsa.target_fire_col,
        CASE WHEN wsa.suppressant > 0
             THEN (SELECT COUNT(*) FROM wildfire_step_fires wsf
                   WHERE wsf.step_id = wsa.step_id
                     AND wsf.is_lit = 1
                     AND ABS(wsf.row - wsa.agent_row) + ABS(wsf.col - wsa.agent_col) <= 1)
             ELSE 0
        END AS n_fightable
    FROM wildfire_step_agents wsa
    JOIN scope s ON s.step_id = wsa.step_id
),
fire_crowd AS (
    SELECT step_id, target_fire_row, target_fire_col, COUNT(*) AS n_fighters
    FROM wildfire_step_agents
    WHERE action_type = 'fight'
    GROUP BY step_id, target_fire_row, target_fire_col
)
SELECT
    f.agent_id,
    COUNT(*)                                                                AS steps,
    SUM(CASE WHEN f.action_type='fight' THEN 1 ELSE 0 END)                 AS fights,
    ROUND(100.0 * SUM(CASE WHEN f.action_type='fight' THEN 1 ELSE 0 END)
          / COUNT(*), 1)                                                    AS "fight%",
    SUM(CASE WHEN f.action_type='noop' AND f.n_fightable = 0 THEN 1 ELSE 0 END)  AS noop_0_fightable,
    SUM(CASE WHEN f.action_type='noop' AND f.n_fightable = 1 THEN 1 ELSE 0 END)  AS noop_1_fightable,
    SUM(CASE WHEN f.action_type='noop' AND f.n_fightable >= 2 THEN 1 ELSE 0 END) AS noop_2plus,
    SUM(CASE WHEN f.action_type='noop' AND f.suppressant <= 0 THEN 1 ELSE 0 END) AS noop_dry,
    ROUND(AVG(f.agent_reward)::numeric, 4)                                  AS avg_rwrd,
    ROUND(100.0 * SUM(CASE WHEN f.action_type='fight' AND fc.n_fighters > 1
                       THEN 1 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN f.action_type='fight' THEN 1 ELSE 0 END), 0),
          1)                                                                AS "coord%"
FROM fightable f
LEFT JOIN fire_crowd fc
    ON f.step_id = fc.step_id
   AND f.target_fire_row = fc.target_fire_row
   AND f.target_fire_col = fc.target_fire_col
   AND f.action_type = 'fight'
GROUP BY f.agent_id
ORDER BY f.agent_id;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q2a  FIRE MACRO — by fire_type                                         │
-- └──────────────────────────────────────────────────────────────────────────┘

WITH scope_ep AS (
    SELECT we.id AS episode_id
    FROM wildfire_episodes we
    JOIN eval_runs evr    ON evr.id = we.eval_run_id
    JOIN eval_results er  ON er.id = evr.eval_id
    WHERE er.experiment_id = :exp_id
),
fire_life AS (
    SELECT
        ws.episode_id, wsf.row, wsf.col,
        MAX(wsf.fire_type)                          AS fire_type,
        MIN(ws.step)                                AS born,
        MAX(ws.step)                                AS last_seen,
        MAX(ws.step) - MIN(ws.step) + 1             AS lifetime,
        MAX(wsf.intensity)                          AS peak_int
    FROM wildfire_step_fires wsf
    JOIN wildfire_steps ws ON ws.id = wsf.step_id
    JOIN scope_ep se       ON se.episode_id = ws.episode_id
    WHERE wsf.is_lit = 1
    GROUP BY ws.episode_id, wsf.row, wsf.col
),
fire_targeted AS (
    SELECT
        ws.episode_id,
        wsa.target_fire_row AS row, wsa.target_fire_col AS col,
        COUNT(DISTINCT wsa.agent_id) AS n_agents,
        COUNT(*)                     AS n_actions
    FROM wildfire_step_agents wsa
    JOIN wildfire_steps ws ON ws.id = wsa.step_id
    JOIN scope_ep se       ON se.episode_id = ws.episode_id
    WHERE wsa.action_type = 'fight'
    GROUP BY ws.episode_id, wsa.target_fire_row, wsa.target_fire_col
),
peak AS (
    SELECT
        ws.episode_id, sub.target_fire_row AS row, sub.target_fire_col AS col,
        MAX(cnt) AS peak_crowd
    FROM (
        SELECT step_id, target_fire_row, target_fire_col, COUNT(*) AS cnt
        FROM wildfire_step_agents WHERE action_type = 'fight'
        GROUP BY step_id, target_fire_row, target_fire_col
    ) sub
    JOIN wildfire_steps ws ON ws.id = sub.step_id
    JOIN scope_ep se       ON se.episode_id = ws.episode_id
    GROUP BY ws.episode_id, sub.target_fire_row, sub.target_fire_col
)
SELECT
    fl.fire_type,
    COUNT(*)                                                     AS n_fires,
    ROUND(AVG(fl.lifetime)::numeric, 1)                          AS avg_life,
    MAX(fl.last_seen)                                            AS last_alive,
    SUM(CASE WHEN ft.row IS NOT NULL THEN 1 ELSE 0 END)         AS touched,
    SUM(CASE WHEN ft.row IS NULL     THEN 1 ELSE 0 END)         AS untouched,
    ROUND(AVG(ft.n_agents)::numeric, 1)                          AS avg_fighters,
    MAX(pk.peak_crowd)                                           AS peak_crowd,
    ROUND(AVG(fl.peak_int)::numeric, 2)                          AS avg_peak_int
FROM fire_life fl
LEFT JOIN fire_targeted ft
    ON fl.episode_id = ft.episode_id AND fl.row = ft.row AND fl.col = ft.col
LEFT JOIN peak pk
    ON fl.episode_id = pk.episode_id AND fl.row = pk.row AND fl.col = pk.col
GROUP BY fl.fire_type
ORDER BY fl.fire_type;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q2b  FIRE MICRO — per (episode, row, col) fire instance               │
-- └──────────────────────────────────────────────────────────────────────────┘

WITH scope_ep AS (
    SELECT we.id AS episode_id
    FROM wildfire_episodes we
    JOIN eval_runs evr    ON evr.id = we.eval_run_id
    JOIN eval_results er  ON er.id = evr.eval_id
    WHERE er.experiment_id = :exp_id
),
fire_life AS (
    SELECT
        ws.episode_id,
        wsf.row AS r, wsf.col AS c,
        MAX(wsf.fire_type)              AS fire_type,
        MIN(ws.step)                    AS born,
        MAX(ws.step)                    AS last_seen,
        MAX(ws.step) - MIN(ws.step) + 1 AS lifetime,
        MAX(wsf.intensity)              AS peak_int
    FROM wildfire_step_fires wsf
    JOIN wildfire_steps ws ON ws.id = wsf.step_id
    JOIN scope_ep se       ON se.episode_id = ws.episode_id
    WHERE wsf.is_lit = 1
    GROUP BY ws.episode_id, wsf.row, wsf.col
),
fire_agents AS (
    SELECT
        ws.episode_id,
        wsa.target_fire_row AS r, wsa.target_fire_col AS c,
        COUNT(DISTINCT wsa.agent_id)         AS unique_agents,
        COUNT(*)                             AS total_actions,
        STRING_AGG(DISTINCT wsa.agent_id, ',') AS agents_involved,
        MAX(sc.step_cnt)                     AS peak_simultaneous
    FROM wildfire_step_agents wsa
    JOIN wildfire_steps ws ON ws.id = wsa.step_id
    JOIN scope_ep se       ON se.episode_id = ws.episode_id
    JOIN (
        SELECT step_id, target_fire_row, target_fire_col, COUNT(*) AS step_cnt
        FROM wildfire_step_agents WHERE action_type = 'fight'
        GROUP BY step_id, target_fire_row, target_fire_col
    ) sc ON sc.step_id = wsa.step_id
        AND sc.target_fire_row = wsa.target_fire_row
        AND sc.target_fire_col = wsa.target_fire_col
    WHERE wsa.action_type = 'fight'
    GROUP BY ws.episode_id, wsa.target_fire_row, wsa.target_fire_col
)
SELECT
    fl.episode_id,
    '(' || fl.r || ',' || fl.c || ')'       AS cell,
    fl.fire_type,
    fl.born,
    fl.last_seen,
    fl.lifetime,
    fl.peak_int,
    COALESCE(fa.unique_agents, 0)            AS agents_fought,
    COALESCE(fa.total_actions, 0)            AS fight_actions,
    COALESCE(fa.peak_simultaneous, 0)        AS peak_simul,
    COALESCE(fa.agents_involved, '—')        AS who,
    CASE WHEN fa.r IS NULL THEN 'IGNORED' ELSE 'FOUGHT' END AS fate
FROM fire_life fl
LEFT JOIN fire_agents fa
    ON fl.episode_id = fa.episode_id AND fl.r = fa.r AND fl.c = fa.c
ORDER BY fl.episode_id, fl.born, fl.r, fl.c;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q3a  COORDINATION MATRIX — agent pairs that co-fight                   │
-- └──────────────────────────────────────────────────────────────────────────┘

SELECT
    a1.agent_id                              AS agent_a,
    a2.agent_id                              AS agent_b,
    COUNT(*)                                 AS co_fights,
    COUNT(DISTINCT a1.target_fire_row||','||a1.target_fire_col) AS shared_fires,
    COUNT(DISTINCT a1.step_id)               AS shared_steps,
    ROUND(AVG(a1.agent_reward + a2.agent_reward)::numeric, 4) AS avg_joint_reward
FROM wildfire_step_agents a1
JOIN wildfire_step_agents a2
    ON  a1.step_id = a2.step_id
    AND a1.agent_id < a2.agent_id
    AND a1.action_type = 'fight'
    AND a2.action_type = 'fight'
    AND a1.target_fire_row = a2.target_fire_row
    AND a1.target_fire_col = a2.target_fire_col
JOIN wildfire_steps ws ON ws.id = a1.step_id
JOIN wildfire_episodes we ON we.id = ws.episode_id
JOIN eval_runs evr ON evr.id = we.eval_run_id
JOIN eval_results er ON er.id = evr.eval_id
WHERE er.experiment_id = :exp_id
GROUP BY a1.agent_id, a2.agent_id
ORDER BY co_fights DESC;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q3b  COORDINATION TARGET — which fire cells drew teamwork              │
-- └──────────────────────────────────────────────────────────────────────────┘
-- One row per fire cell that had ≥2 agents at the same step at least once.
-- avg_coord_reward vs avg_solo_reward: did teaming up help?

WITH coord_events AS (
    SELECT
        ws.episode_id,
        wsa.target_fire_row AS r, wsa.target_fire_col AS c,
        wsa.step_id,
        COUNT(DISTINCT wsa.agent_id)    AS n_agents,
        AVG(wsa.agent_reward)           AS step_avg_reward
    FROM wildfire_step_agents wsa
    JOIN wildfire_steps ws ON ws.id = wsa.step_id
    JOIN wildfire_episodes we ON we.id = ws.episode_id
    JOIN eval_runs evr ON evr.id = we.eval_run_id
    JOIN eval_results er ON er.id = evr.eval_id
    WHERE er.experiment_id = :exp_id
      AND wsa.action_type = 'fight'
    GROUP BY ws.episode_id, wsa.target_fire_row, wsa.target_fire_col, wsa.step_id
    HAVING COUNT(DISTINCT wsa.agent_id) >= 2
),
coord_agents AS (
    SELECT
        ws.episode_id,
        wsa.target_fire_row AS r, wsa.target_fire_col AS c,
        STRING_AGG(DISTINCT wsa.agent_id, ',') AS agents
    FROM wildfire_step_agents wsa
    JOIN wildfire_steps ws ON ws.id = wsa.step_id
    JOIN coord_events ce
        ON ce.episode_id = ws.episode_id
       AND ce.r = wsa.target_fire_row
       AND ce.c = wsa.target_fire_col
       AND ce.step_id = wsa.step_id
    WHERE wsa.action_type = 'fight'
    GROUP BY ws.episode_id, wsa.target_fire_row, wsa.target_fire_col
),
fire_type_lookup AS (
    SELECT DISTINCT ON (ws.episode_id, wsf.row, wsf.col)
        ws.episode_id, wsf.row, wsf.col, wsf.fire_type
    FROM wildfire_step_fires wsf
    JOIN wildfire_steps ws ON ws.id = wsf.step_id
    WHERE wsf.is_lit = 1
)
SELECT
    ce.episode_id,
    '(' || ce.r || ',' || ce.c || ')'              AS cell,
    ftl.fire_type,
    COUNT(*)                                        AS coord_steps,
    MAX(ce.n_agents)                                AS max_simul,
    ca.agents                                       AS who,
    ROUND(AVG(ce.step_avg_reward)::numeric, 4)      AS avg_coord_reward,
    ROUND((
        SELECT AVG(wsa2.agent_reward)
        FROM wildfire_step_agents wsa2
        JOIN wildfire_steps ws2 ON ws2.id = wsa2.step_id
        WHERE ws2.episode_id = ce.episode_id
          AND wsa2.target_fire_row = ce.r
          AND wsa2.target_fire_col = ce.c
          AND wsa2.action_type = 'fight'
          AND wsa2.step_id NOT IN (SELECT step_id FROM coord_events ce2
                                   WHERE ce2.episode_id = ce.episode_id
                                     AND ce2.r = ce.r AND ce2.c = ce.c)
    )::numeric, 4)                                  AS avg_solo_reward
FROM coord_events ce
JOIN coord_agents ca
    ON ce.episode_id = ca.episode_id AND ce.r = ca.r AND ce.c = ca.c
LEFT JOIN fire_type_lookup ftl
    ON ftl.episode_id = ce.episode_id AND ftl.row = ce.r AND ftl.col = ce.c
GROUP BY ce.episode_id, ce.r, ce.c, ca.agents, ftl.fire_type
ORDER BY coord_steps DESC;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q4  NOOP CONTEXT — fire pressure when agents chose NOOP                │
-- └──────────────────────────────────────────────────────────────────────────┘

SELECT
    CASE
        WHEN ws.active_fire_count = 0 THEN '0 fires'
        WHEN ws.active_fire_count = 1 THEN '1 fire'
        WHEN ws.active_fire_count = 2 THEN '2 fires'
        WHEN ws.active_fire_count <= 4 THEN '3-4 fires'
        ELSE '5+ fires'
    END                                         AS fire_pressure,
    COUNT(*)                                    AS noop_count,
    ROUND(AVG(wsa.suppressant)::numeric, 2)     AS avg_suppressant,
    ROUND(AVG(wsa.agent_reward)::numeric, 4)    AS avg_reward,
    ROUND(AVG(ws.total_intensity_sum)::numeric, 2) AS avg_total_intensity,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS "pct_of_noops"
FROM wildfire_step_agents wsa
JOIN wildfire_steps ws     ON ws.id = wsa.step_id
JOIN wildfire_episodes we  ON we.id = ws.episode_id
JOIN eval_runs evr         ON evr.id = we.eval_run_id
JOIN eval_results er       ON er.id = evr.eval_id
WHERE er.experiment_id = :exp_id
  AND wsa.action_type = 'noop'
GROUP BY fire_pressure, ws.active_fire_count
ORDER BY ws.active_fire_count;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q5  FIRE PRESSURE TIMELINE — per timestep across episodes              │
-- └──────────────────────────────────────────────────────────────────────────┘

SELECT
    ws.step,
    ROUND(AVG(ws.active_fire_count)::numeric, 1)       AS avg_fires,
    ROUND(AVG(ws.total_intensity_sum)::numeric, 1)     AS avg_intensity,
    ROUND(AVG(ws.new_burnouts)::numeric, 2)            AS avg_burnouts,
    ROUND(AVG(ws.new_putouts)::numeric, 2)             AS avg_putouts,
    SUM(CASE WHEN wsa.action_type='fight' THEN 1 ELSE 0 END) AS fights,
    SUM(CASE WHEN wsa.action_type='noop'  THEN 1 ELSE 0 END) AS noops,
    ROUND(100.0 * SUM(CASE WHEN wsa.action_type='fight' THEN 1 ELSE 0 END)
          / NULLIF(COUNT(wsa.id), 0), 1)               AS "fight%"
FROM wildfire_steps ws
JOIN wildfire_episodes we  ON we.id = ws.episode_id
JOIN eval_runs evr         ON evr.id = we.eval_run_id
JOIN eval_results er       ON er.id = evr.eval_id
LEFT JOIN wildfire_step_agents wsa ON wsa.step_id = ws.id
WHERE er.experiment_id = :exp_id
GROUP BY ws.step
ORDER BY ws.step;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q6  EPISODE SUMMARY — correct per-agent fight rate                     │
-- └──────────────────────────────────────────────────────────────────────────┘

WITH agent_counts AS (
    SELECT ws.episode_id, COUNT(DISTINCT wsa.agent_id) AS n_agents
    FROM wildfire_step_agents wsa
    JOIN wildfire_steps ws ON ws.id = wsa.step_id
    GROUP BY ws.episode_id
)
SELECT
    we.id                                                       AS ep_id,
    we.seed,
    we.step_count,
    ac.n_agents,
    we.total_reward,
    we.burnouts,
    we.putouts,
    we.total_fires,
    we.fight_count,
    we.noop_count,
    ROUND(1.0 * we.fight_count
          / NULLIF(we.step_count * ac.n_agents, 0), 3)          AS fight_rate,
    ROUND(1.0 * we.putouts / NULLIF(we.total_fires, 0), 2)     AS putout_rate,
    ROUND(1.0 * we.burnouts / NULLIF(we.total_fires, 0), 2)    AS burnout_rate,
    (SELECT MAX(ws2.step)
     FROM wildfire_steps ws2
     WHERE ws2.episode_id = we.id AND ws2.active_fire_count > 0
    )                                                           AS last_fire_step,
    CASE WHEN (
        SELECT ws3.active_fire_count
        FROM wildfire_steps ws3
        WHERE ws3.episode_id = we.id
        ORDER BY ws3.step DESC LIMIT 1
    ) = 0 THEN 1 ELSE 0 END                                    AS all_clear
FROM wildfire_episodes we
JOIN eval_runs evr    ON evr.id = we.eval_run_id
JOIN eval_results er  ON er.id = evr.eval_id
JOIN agent_counts ac  ON ac.episode_id = we.id
WHERE er.experiment_id = :exp_id
ORDER BY we.total_reward DESC;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q7  THE IGNORED — fires nobody ever fought, grouped by type            │
-- └──────────────────────────────────────────────────────────────────────────┘

WITH all_fires AS (
    SELECT DISTINCT ws.episode_id, wsf.row, wsf.col, wsf.fire_type
    FROM wildfire_step_fires wsf
    JOIN wildfire_steps ws ON ws.id = wsf.step_id
    JOIN wildfire_episodes we ON we.id = ws.episode_id
    JOIN eval_runs evr ON evr.id = we.eval_run_id
    JOIN eval_results er ON er.id = evr.eval_id
    WHERE wsf.is_lit = 1
      AND er.experiment_id = :exp_id
),
fought_fires AS (
    SELECT DISTINCT ws.episode_id, wsa.target_fire_row AS row, wsa.target_fire_col AS col
    FROM wildfire_step_agents wsa
    JOIN wildfire_steps ws ON ws.id = wsa.step_id
    JOIN wildfire_episodes we ON we.id = ws.episode_id
    JOIN eval_runs evr ON evr.id = we.eval_run_id
    JOIN eval_results er ON er.id = evr.eval_id
    WHERE wsa.action_type = 'fight'
      AND er.experiment_id = :exp_id
)
SELECT
    af.fire_type,
    COUNT(*)                                                     AS total_fires,
    SUM(CASE WHEN ff.row IS NULL THEN 1 ELSE 0 END)             AS ignored,
    ROUND(100.0 * SUM(CASE WHEN ff.row IS NULL THEN 1 ELSE 0 END)
          / COUNT(*), 1)                                         AS "ignored%"
FROM all_fires af
LEFT JOIN fought_fires ff
    ON af.episode_id = ff.episode_id AND af.row = ff.row AND af.col = ff.col
GROUP BY af.fire_type
ORDER BY af.fire_type;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q8  AGENT × FIRE TYPE MATRIX — fight distribution per agent per type   │
-- └──────────────────────────────────────────────────────────────────────────┘

SELECT
    wsa.agent_id,
    wsf.fire_type,
    COUNT(*)                                                     AS fights_on_type,
    ROUND(100.0 * COUNT(*)
          / SUM(COUNT(*)) OVER (PARTITION BY wsa.agent_id), 1)   AS "pct%",
    ROUND(AVG(wsa.agent_reward)::numeric, 4)                     AS avg_rwrd,
    ROUND(AVG(wsf.intensity)::numeric, 2)                        AS avg_int_targeted
FROM wildfire_step_agents wsa
JOIN wildfire_steps ws     ON ws.id = wsa.step_id
JOIN wildfire_episodes we  ON we.id = ws.episode_id
JOIN eval_runs evr         ON evr.id = we.eval_run_id
JOIN eval_results er       ON er.id = evr.eval_id
JOIN wildfire_step_fires wsf
    ON  wsf.step_id = wsa.step_id
    AND wsf.row = wsa.target_fire_row
    AND wsf.col = wsa.target_fire_col
WHERE er.experiment_id = :exp_id
  AND wsa.action_type = 'fight'
GROUP BY wsa.agent_id, wsf.fire_type
ORDER BY wsa.agent_id, fights_on_type DESC;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q9  AGENT → TOP FIRE CELLS — top-3 most-attacked positions per agent   │
-- └──────────────────────────────────────────────────────────────────────────┘

WITH agent_fire_cells AS (
    SELECT
        wsa.agent_id,
        ws.episode_id,
        wsa.target_fire_row AS r, wsa.target_fire_col AS c,
        MAX(wsf.fire_type)                  AS fire_type,
        COUNT(*)                            AS fights,
        ROUND(AVG(wsa.agent_reward)::numeric, 4)  AS avg_rwrd,
        ROUND(AVG(wsf.intensity)::numeric, 2)      AS avg_int
    FROM wildfire_step_agents wsa
    JOIN wildfire_steps ws ON ws.id = wsa.step_id
    JOIN wildfire_episodes we ON we.id = ws.episode_id
    JOIN eval_runs evr ON evr.id = we.eval_run_id
    JOIN eval_results er ON er.id = evr.eval_id
    JOIN wildfire_step_fires wsf
        ON  wsf.step_id = wsa.step_id
        AND wsf.row = wsa.target_fire_row
        AND wsf.col = wsa.target_fire_col
    WHERE er.experiment_id = :exp_id
      AND wsa.action_type = 'fight'
    GROUP BY wsa.agent_id, ws.episode_id, wsa.target_fire_row, wsa.target_fire_col
),
ranked AS (
    SELECT
        agent_id, episode_id,
        '(' || r || ',' || c || ')' AS cell,
        fire_type, fights,
        ROUND(100.0 * fights / SUM(fights) OVER (PARTITION BY agent_id, episode_id), 1) AS "share%",
        avg_rwrd, avg_int,
        ROW_NUMBER() OVER (
            PARTITION BY agent_id, episode_id ORDER BY fights DESC
        ) AS rk
    FROM agent_fire_cells
)
SELECT agent_id, episode_id, cell, fire_type, fights, "share%", avg_rwrd, avg_int
FROM ranked
WHERE rk <= 3
ORDER BY agent_id, episode_id, rk;


-- ┌──────────────────────────────────────────────────────────────────────────┐
-- │ Q10  AGENT FIGHT TIMELINE — early / mid / late phase behavior          │
-- └──────────────────────────────────────────────────────────────────────────┘

WITH ep_len AS (
    SELECT we.id AS episode_id, we.step_count
    FROM wildfire_episodes we
    JOIN eval_runs evr ON evr.id = we.eval_run_id
    JOIN eval_results er ON er.id = evr.eval_id
    WHERE er.experiment_id = :exp_id
)
SELECT
    wsa.agent_id,
    CASE
        WHEN ws.step < el.step_count / 3.0     THEN '1_early'
        WHEN ws.step < 2 * el.step_count / 3.0 THEN '2_mid'
        ELSE                                         '3_late'
    END AS phase,
    SUM(CASE WHEN wsa.action_type='fight' THEN 1 ELSE 0 END) AS fights,
    SUM(CASE WHEN wsa.action_type='noop'  THEN 1 ELSE 0 END) AS noops,
    ROUND(100.0 * SUM(CASE WHEN wsa.action_type='fight' THEN 1 ELSE 0 END)
          / COUNT(*), 1)                                       AS "fight%",
    ROUND(AVG(wsa.agent_reward)::numeric, 4)                   AS avg_rwrd,
    ROUND(AVG(ws.active_fire_count)::numeric, 1)               AS avg_fires
FROM wildfire_step_agents wsa
JOIN wildfire_steps ws ON ws.id = wsa.step_id
JOIN ep_len el         ON el.episode_id = ws.episode_id
WHERE ws.id IN (
    SELECT ws2.id FROM wildfire_steps ws2
    JOIN wildfire_episodes we2 ON we2.id = ws2.episode_id
    JOIN eval_runs evr2 ON evr2.id = we2.eval_run_id
    JOIN eval_results er2 ON er2.id = evr2.eval_id
    WHERE er2.experiment_id = :exp_id
)
GROUP BY wsa.agent_id, phase
ORDER BY wsa.agent_id, phase;