PRAGMA foreign_keys = ON;

-- Experiments: training or execution, smart or naive
CREATE TABLE IF NOT EXISTS experiments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    run_type                TEXT NOT NULL CHECK (run_type IN ('training', 'execution')),
    model_family            TEXT,      -- "PLATO", "DGN", "DICG", "MOHITO", "weakest", "strongest", "random", "noop"
    model_type              TEXT CHECK (model_type IN ('smart', 'naive')),
    name                    TEXT,      -- e.g. "PLATO", 

    variation1              TEXT,      -- "mlp", "lstm", "gru"
    variation2              TEXT,      -- "dotproduct", "additive"

    grid_size               INTEGER CHECK (grid_size > 0),
    setting_type            TEXT,     
    seed                    INTEGER,

    parent_experiment_id    INTEGER,   -- NULL for main training runs

    checkpoint_path         TEXT,
    zero_shot  INTEGER CHECK (zero_shot IN (0, 1)),
    zero_shot_source_grids_size  INTEGER,
    zero_shot_target_grids_size  INTEGER,
    zero_shot_target_grids_size_detail TEXT,
    zero_shot_mode TEXT CHECK (zero_shot_mode IS NULL OR zero_shot_mode IN ('fixteam', 'fixteam_transition', 'zs', 'bigteam')),
    notes                   TEXT,
    domain                  TEXT CHECK (domain IS NULL OR domain IN ('wildfire', 'rideshare')),

    FOREIGN KEY (parent_experiment_id) REFERENCES experiments(id)
);

-- Per training or execution episode
CREATE TABLE IF NOT EXISTS episode_results (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at                  DATETIME DEFAULT CURRENT_TIMESTAMP,
    experiment_id               INTEGER NOT NULL,
    episode                     INTEGER NOT NULL CHECK (episode >= 0),

    -- core episode stats
    reward                      REAL,
    episode_length              INTEGER,

    -- wildfire metrics
    burnouts                    REAL,
    putouts                     REAL,
    episode_total_fires_count   REAL,
    new_fire_ignited_count      REAL,
    episode_fight_count         REAL,


    -- rideshare metrics
    pickups                     REAL,
    dropoffs                    REAL,
    episode_total_tasks         REAL,
    new_tasks_completed         REAL,
    episode_accept_count        REAL,
    episode_pick_count          REAL,
    episode_drop_count          REAL,



    -- shared action metrics
    episode_noop_count          REAL,

    -- PPO / loss metrics (NULL for pure execution / naive)
    actor_loss                  REAL,
    critic_loss                 REAL,
    entropy                     REAL,
    approx_kl                   REAL,
    value_mse                   REAL,
    explained_variance          REAL,
    main_loss                   REAL,

    -- convergence and gradient info
    total_actor_grad            REAL,
    total_critic_grad           REAL,
    actor_vanishing             INTEGER,
    actor_exploding             INTEGER,
    critic_vanishing            INTEGER,
    critic_exploding            INTEGER,

    -- system metrics
    ram_usage_mb                REAL,
    episode_seconds             REAL,
    steps_per_second            REAL,
    env_seeds                   TEXT,
    episode_supressant_wait_time REAL,


    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Aggregated evaluation over many runs
CREATE TABLE IF NOT EXISTS eval_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    experiment_id   INTEGER NOT NULL,
    episode         INTEGER,     -- training episode, or NULL for pure execution
    avg_reward      REAL,
    avg_burnouts    REAL,
    avg_putouts     REAL,
    avg_dropoffs    REAL,
    avg_pickups     REAL,  
    avg_dropoffs_success REAL,  
    avg_accepts     REAL,
    avg_noop        REAL,
    avg_total_actions       REAL,
    avg_length      REAL,
    avg_reward_per_fight REAL,
    avg_putouts_per_fight REAL,
    avg_zero_suppressant_count REAL,

    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Individual runs inside one eval_results (e.g. 30 runs)
CREATE TABLE IF NOT EXISTS eval_runs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at                  DATETIME DEFAULT CURRENT_TIMESTAMP,
    eval_id                     INTEGER NOT NULL,
    run_index                   INTEGER NOT NULL CHECK (run_index >= 0),
    seed                        INTEGER,
    eval_seeds                  TEXT,
    reward                      REAL,
    episode_length              REAL,

    -- wildfire metrics
    burnouts                    REAL,
    putouts                     REAL,
    episode_total_fires_count   REAL,
    new_fire_ignited_count      REAL,
    episode_fight_count         REAL,

    -- rideshare metrics
    pickups                     REAL,
    dropoffs                    REAL,
    episode_total_tasks         REAL,
    new_tasks_completed         REAL,
    episode_accept_count        REAL,
    episode_pick_count          REAL,
    episode_drop_count          REAL,

    episode_noop_count          REAL,
    reward_per_fight            REAL,
    putouts_per_fight           REAL,
    episode_zero_suppressant_count REAL,

    FOREIGN KEY (eval_id) REFERENCES eval_results(id)
);



-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_episode_results_experiment_id
    ON episode_results (experiment_id);

CREATE INDEX IF NOT EXISTS idx_eval_results_experiment_id
    ON eval_results (experiment_id);

CREATE INDEX IF NOT EXISTS idx_eval_runs_eval_id
    ON eval_runs (eval_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- Wildfire detailed step tracking (wildfire domain only)
-- ─────────────────────────────────────────────────────────────────────────────

-- One row per episode inside an eval_run (wildfire only)
CREATE TABLE IF NOT EXISTS wildfire_episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    eval_run_id     INTEGER NOT NULL,       -- FK → eval_runs.id
    seed            INTEGER,
    step_count      INTEGER,
    total_reward    REAL,
    burnouts        REAL,
    putouts         REAL,
    total_fires     REAL,
    new_fires       REAL,
    fight_count     INTEGER,
    noop_count      INTEGER,
    FOREIGN KEY (eval_run_id) REFERENCES eval_runs(id)
);

-- One row per timestep inside a wildfire_episode
CREATE TABLE IF NOT EXISTS wildfire_steps (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    episode_id          INTEGER NOT NULL,   -- FK → wildfire_episodes.id
    step                INTEGER NOT NULL,
    step_reward         REAL,               -- mean reward across agents this step
    active_fire_count   INTEGER,            -- lit fires (fires > 0) at start of step
    total_intensity_sum REAL,               -- sum of intensity across all lit fires
    new_burnouts        INTEGER,            -- burnouts that occurred this step
    new_putouts         INTEGER,            -- putouts that occurred this step
    FOREIGN KEY (episode_id) REFERENCES wildfire_episodes(id)
);

-- One row per non-zero fire cell per step (state agents observed)
CREATE TABLE IF NOT EXISTS wildfire_step_fires (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id     INTEGER NOT NULL,           -- FK → wildfire_steps.id
    row         INTEGER NOT NULL,
    col         INTEGER NOT NULL,
    intensity   REAL,
    fuel        REAL,
    is_lit      INTEGER NOT NULL CHECK (is_lit IN (0, 1)),  -- fires > 0
    fire_type   INTEGER,                    -- abs(fires[row,col])
    FOREIGN KEY (step_id) REFERENCES wildfire_steps(id)
);

-- One row per agent per step: position, suppressant, action taken
CREATE TABLE IF NOT EXISTS wildfire_step_agents (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id               INTEGER NOT NULL,   -- FK → wildfire_steps.id
    agent_id              TEXT NOT NULL,
    agent_row             INTEGER,
    agent_col             INTEGER,
    suppressant           REAL,
    capacity              REAL,
    equipment             REAL,
    action_type           TEXT CHECK (action_type IN ('fight', 'noop', 'unknown')),
    target_fire_row       INTEGER,            -- NULL if noop/unknown
    target_fire_col       INTEGER,            -- NULL if noop/unknown
    target_fire_intensity REAL,               -- NULL if noop/unknown
    target_fire_fuel      REAL,               -- NULL if noop/unknown
    agent_reward          REAL,               -- per-agent reward this step (post-step)
    FOREIGN KEY (step_id) REFERENCES wildfire_steps(id)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_wildfire_episodes_eval_run_id
    ON wildfire_episodes (eval_run_id);

CREATE INDEX IF NOT EXISTS idx_wildfire_steps_episode_id
    ON wildfire_steps (episode_id);

CREATE INDEX IF NOT EXISTS idx_wildfire_step_fires_step_id
    ON wildfire_step_fires (step_id);

CREATE INDEX IF NOT EXISTS idx_wildfire_step_agents_step_id
    ON wildfire_step_agents (step_id);

CREATE INDEX IF NOT EXISTS idx_wildfire_step_agents_agent_id
    ON wildfire_step_agents (agent_id);

CREATE INDEX IF NOT EXISTS idx_wildfire_step_agents_action_type
    ON wildfire_step_agents (action_type);

