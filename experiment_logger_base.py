"""
Shared base classes for SQLite and PostgreSQL experiment loggers.
Backend-specific files (experiment_logger_sqlite.py, experiment_logger_postgres.py)
subclass BaseExperimentDB and set RunLogger = BaseRunLogger.
"""
import dataset
import numpy as np
import logging
import psutil
import tabulate
import numbers


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def _to_native(v):
    """Convert numpy scalar Python scalar."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, numbers.Number):
        return float(v) if isinstance(v, float) else v
    return v


def _normalize_dict(d: dict) -> dict:
    return {k: _to_native(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# BaseExperimentDB
# ---------------------------------------------------------------------------

class BaseExperimentDB:
    """Backend-agnostic DB layer.  Subclasses must implement _init_db()."""

    def __init__(self, url: str):
        self._init_db()
        self.db = dataset.connect(url)
        self.experiments = self.db["experiments"]
        self.episodes = self.db["episode_results"]
        self.evals = self.db["eval_results"]
        self.eval_runs = self.db["eval_runs"]
        self.convergence = self.db["convergence_results"]
        # Wildfire detailed step tracking tables
        self.wf_episodes = self.db["wildfire_episodes"]
        self.wf_steps = self.db["wildfire_steps"]
        self.wf_step_fires = self.db["wildfire_step_fires"]
        self.wf_step_agents = self.db["wildfire_step_agents"]

    # -- hooks ---------------------------------------------------------------

    def _init_db(self):
        """Called before dataset.connect().  Override to create/migrate schema."""
        pass

    def _after_log_execution(self):
        """Called at the end of log_execution().  Override for e.g. explicit COMMIT."""
        pass

    # -- write helpers -------------------------------------------------------

    def create_experiment(self, meta: dict) -> int:
        return self.experiments.insert(meta)

    def log_episode(self, experiment_id: int, episode: int, row: dict) -> None:
        full = _normalize_dict(dict(experiment_id=experiment_id, episode=episode, **row))
        self.episodes.insert(full)

    def log_execution(self, experiment_meta, eval_stats, eval_runs,
                      convergence_results=None, wildfire_step_data_list=None):
        """Insert experiment + eval + per-run rows in one call."""
        experiment_meta = _normalize_dict(experiment_meta)
        eval_stats = _normalize_dict(eval_stats)

        experiment_id = self.experiments.insert(experiment_meta)
        eval_id = self.evals.insert({
            'experiment_id': experiment_id,
            'episode': None,
            **eval_stats,
        })

        for i, run in enumerate(eval_runs):
            normalized_run = _normalize_dict(run)
            run_id = self.eval_runs.insert({
                'eval_id': eval_id,
                'run_index': i,
                **normalized_run,
            })
            if wildfire_step_data_list is not None and i < len(wildfire_step_data_list):
                step_data = wildfire_step_data_list[i]
                if step_data is not None:
                    self._log_wildfire_episode(run_id, run, step_data)

        if convergence_results:
            for agent, stats in convergence_results.items():
                self.convergence.insert({
                    'eval_id': eval_id,
                    'agent_name': agent,
                    **_normalize_dict(stats),
                })

        self._after_log_execution()
        return eval_id

    def _log_wildfire_episode(self, eval_run_id: int, run_record: dict, step_data: list):
        """Insert wildfire_episodes + all child steps/fires/agents rows."""
        episode_id = self.wf_episodes.insert({
            'eval_run_id': eval_run_id,
            'seed': run_record.get('seed'),
            'step_count': run_record.get('episode_length'),
            'total_reward': _to_native(run_record.get('reward')),
            'burnouts': _to_native(run_record.get('burnouts')),
            'putouts': _to_native(run_record.get('putouts')),
            'total_fires': _to_native(run_record.get('episode_total_fires_count')),
            'new_fires': _to_native(run_record.get('new_fire_ignited_count')),
            'fight_count': run_record.get('episode_fight_count'),
            'noop_count': run_record.get('episode_noop_count'),
        })

        for step_record in step_data:
            step_id = self.wf_steps.insert({
                'episode_id': episode_id,
                'step': step_record['step'],
                'step_reward': _to_native(step_record['step_reward']),
                'active_fire_count': step_record['active_fire_count'],
                'total_intensity_sum': _to_native(step_record['total_intensity_sum']),
                'new_burnouts': step_record.get('new_burnouts', 0),
                'new_putouts': step_record.get('new_putouts', 0),
            })

            fires = step_record.get('fires', [])
            if fires:
                self.wf_step_fires.insert_many(
                    [{'step_id': step_id, **f} for f in fires]
                )

            agents = step_record.get('agents', [])
            if agents:
                self.wf_step_agents.insert_many(
                    [{'step_id': step_id, **a} for a in agents]
                )

    def create_eval(self, experiment_id: int, episode) -> int:
        row = dict(
            experiment_id=experiment_id,
            episode=episode,
            avg_reward=None,
            avg_burnouts=None,
            avg_putouts=None,
            avg_pickups=None,
            avg_dropoffs=None,
            avg_dropoffs_success=None,
            avg_accepts=None,
            avg_noop=None,
            avg_length=None,
            avg_convergence=None,
        )
        return self.evals.insert(row)

    def update_eval(self, eval_id: int, stats: dict) -> None:
        self.evals.update(dict(id=eval_id, **_normalize_dict(stats)), ["id"])

    def log_eval_run(self, eval_id: int, run_index: int, row: dict) -> None:
        full = _normalize_dict(dict(eval_id=eval_id, run_index=run_index, **row))
        self.eval_runs.insert(full)

    def log_convergence(self, eval_id: int, agent_name: str, stats: dict) -> None:
        full = _normalize_dict(dict(eval_id=eval_id, agent_name=agent_name, **stats))
        self.convergence.insert(full)


# ---------------------------------------------------------------------------
# BaseRunLogger
# ---------------------------------------------------------------------------

class BaseRunLogger:
    """TensorBoard + DB + console logging. Works with any BaseExperimentDB subclass."""

    def __init__(
        self,
        writer,
        db: BaseExperimentDB,
        experiment_meta: dict,
        log_every: int,
    ):
        self.writer = writer
        self.db = db
        self.log_every = log_every
        self.experiment_id = self.db.create_experiment(experiment_meta)

    def log_episode(
        self,
        model_name: str,
        episode: int,
        episode_stats: dict,
        train_metrics: dict,
        grad_summary: dict,
        total_actor_grad: float,
        total_critic_grad: float,
        elapsed: float,
        fps: float,
    ):
        reward = episode_stats.get("reward", 0)
        if self.writer:
            w = self.writer
            w.add_scalar("General/Reward", reward, episode)

            # Wildfire metrics
            w.add_scalar("General/burnouts", episode_stats.get("burnouts", 0), episode)
            w.add_scalar("General/putouts", episode_stats.get("putouts", 0), episode)
            w.add_scalar("General/episode_total_fires_count",
                         episode_stats.get("episode_total_fires_count", 0), episode)
            w.add_scalar("General/new_fire_ignited_count",
                         episode_stats.get("new_fire_ignited_count", 0), episode)
            w.add_scalar("General/episode_fight_count",
                         episode_stats.get("episode_fight_count", 0), episode)

            # Rideshare metrics
            w.add_scalar("General/pickups", episode_stats.get("pickups", 0), episode)
            w.add_scalar("General/dropoffs", episode_stats.get("dropoffs", 0), episode)
            w.add_scalar("General/episode_total_tasks",
                         episode_stats.get("episode_total_tasks", 0), episode)
            w.add_scalar("General/new_tasks_completed",
                         episode_stats.get("new_tasks_completed", 0), episode)
            w.add_scalar("General/episode_accept_count",
                         episode_stats.get("episode_accept_count", 0), episode)
            w.add_scalar("General/episode_pick_count",
                         episode_stats.get("episode_pick_count", 0), episode)
            w.add_scalar("General/episode_drop_count",
                         episode_stats.get("episode_drop_count", 0), episode)

            # Shared metrics
            w.add_scalar("General/episode_noop_count",
                         episode_stats.get("episode_noop_count", 0), episode)

            w.add_scalar(f"{model_name}/aa_Reward", reward, episode)

            # Wildfire metrics
            w.add_scalar(f"{model_name}/burnouts", episode_stats.get("burnouts", 0), episode)
            w.add_scalar(f"{model_name}/putouts", episode_stats.get("putouts", 0), episode)
            w.add_scalar(f"{model_name}/episode_total_fires_count",
                         episode_stats.get("episode_total_fires_count", 0), episode)
            w.add_scalar(f"{model_name}/new_fire_ignited_count",
                         episode_stats.get("new_fire_ignited_count", 0), episode)
            w.add_scalar(f"{model_name}/episode_fight_count",
                         episode_stats.get("episode_fight_count", 0), episode)

            # Rideshare metrics
            w.add_scalar(f"{model_name}/pickups", episode_stats.get("pickups", 0), episode)
            w.add_scalar(f"{model_name}/dropoffs", episode_stats.get("dropoffs", 0), episode)
            w.add_scalar(f"{model_name}/episode_total_tasks",
                         episode_stats.get("episode_total_tasks", 0), episode)
            w.add_scalar(f"{model_name}/new_tasks_completed",
                         episode_stats.get("new_tasks_completed", 0), episode)
            w.add_scalar(f"{model_name}/episode_accept_count",
                         episode_stats.get("episode_accept_count", 0), episode)
            w.add_scalar(f"{model_name}/episode_pick_count",
                         episode_stats.get("episode_pick_count", 0), episode)
            w.add_scalar(f"{model_name}/episode_drop_count",
                         episode_stats.get("episode_drop_count", 0), episode)

            # Shared metrics
            w.add_scalar(f"{model_name}/episode_length",
                         episode_stats.get("episode_length", 0), episode)
            w.add_scalar(f"{model_name}/episode_noop_count",
                         episode_stats.get("episode_noop_count", 0), episode)

            # Gradients and timing
            w.add_scalar(f"{model_name}/total_actor_grad", total_actor_grad, episode)
            w.add_scalar(f"{model_name}/total_critic_grad", total_critic_grad, episode)
            w.add_scalar(f"{model_name}/episode_seconds", elapsed, episode)
            w.add_scalar(f"{model_name}/steps_per_second", fps, episode)

            process = psutil.Process()
            ram_usage_mb = process.memory_info().rss / 1024 / 1024
            if ram_usage_mb is not None:
                w.add_scalar(f"{model_name}/ram_usage", ram_usage_mb, episode)

            # PPO metrics
            for key, value in train_metrics.items():
                w.add_scalar(f"{model_name}/{key}", value, episode)

        # Episode row for DB
        process = psutil.Process()
        ram_usage_mb = process.memory_info().rss / 1024 / 1024
        main_loss = train_metrics.get("actor_loss", 0) + train_metrics.get("critic_loss", 0)

        db_row = dict(
            reward=reward,

            # Wildfire metrics
            burnouts=episode_stats.get("burnouts"),
            putouts=episode_stats.get("putouts"),
            episode_total_fires_count=episode_stats.get("episode_total_fires_count"),
            new_fire_ignited_count=episode_stats.get("new_fire_ignited_count"),
            episode_fight_count=episode_stats.get("episode_fight_count"),

            # Rideshare metrics
            pickups=episode_stats.get("pickups"),
            dropoffs=episode_stats.get("dropoffs"),
            episode_total_tasks=episode_stats.get("episode_total_tasks"),
            new_tasks_completed=episode_stats.get("new_tasks_completed"),
            episode_accept_count=episode_stats.get("episode_accept_count"),
            episode_pick_count=episode_stats.get("episode_pick_count"),
            episode_drop_count=episode_stats.get("episode_drop_count"),
            episode_supressant_wait_time=episode_stats.get("episode_supressant_wait_time"),

            # Shared metrics
            episode_noop_count=episode_stats.get("episode_noop_count"),
            episode_length=episode_stats.get("episode_length"),

            # Training metrics
            actor_loss=train_metrics.get("actor_loss"),
            critic_loss=train_metrics.get("critic_loss"),
            entropy=train_metrics.get("entropy"),
            approx_kl=train_metrics.get("approx_kl"),
            value_mse=train_metrics.get("value_mse"),
            explained_variance=train_metrics.get("explained_variance"),
            main_loss=main_loss,
            avg_convergence=None,
            total_actor_grad=total_actor_grad,
            total_critic_grad=total_critic_grad,
            actor_vanishing=grad_summary.get("actor_vanishing"),
            actor_exploding=grad_summary.get("actor_exploding"),
            critic_vanishing=grad_summary.get("critic_vanishing"),
            critic_exploding=grad_summary.get("critic_exploding"),
            ram_usage_mb=ram_usage_mb,
            episode_seconds=elapsed,
            steps_per_second=fps,
            env_seeds=str(episode_stats.get('env_seeds', '')),
        )
        self.db.log_episode(self.experiment_id, episode, db_row)

        if episode % self.log_every == 0:
            table = [
                ["Episode", episode],
                ["Reward", reward],
                ["Actor Loss", train_metrics.get("actor_loss", 0)],
                ["Actor Grad Norm", total_actor_grad],
                ["Critic Loss", train_metrics.get("critic_loss", 0)],
                ["Critic Grad Norm", total_critic_grad],
                ["Entropy", train_metrics.get("entropy", 0)],
                ["KL Divergence", train_metrics.get("approx_kl", 0)],
                ["Episode Seconds", elapsed],
                ["Steps/sec", fps],
                ["Actor Vanishing", grad_summary.get("actor_vanishing", 0)],
                ["Actor Exploding", grad_summary.get("actor_exploding", 0)],
                ["Critic Vanishing", grad_summary.get("critic_vanishing", 0)],
                ["Critic Exploding", grad_summary.get("critic_exploding", 0)],
            ]
            logging.info(tabulate.tabulate(table, headers=["Name", "Value"], tablefmt="orgtbl"))

    def log_eval(
        self,
        episode,
        eval_reward: float,
        eval_burnouts: float,
        eval_putouts: float,
        avg_pickups: float,
        avg_dropoffs: float,
        avg_dropoffs_success: float,
        avg_accepts: float,
        avg_noop: float,
        eval_length: float,
        avg_convergence,
        convergence_results: dict | None = None,
    ):
        eval_id = self.db.create_eval(self.experiment_id, episode)

        stats = dict(
            avg_reward=eval_reward,
            avg_burnouts=eval_burnouts,
            avg_putouts=eval_putouts,
            avg_pickups=avg_pickups,
            avg_dropoffs=avg_dropoffs,
            avg_dropoffs_success=avg_dropoffs_success,
            avg_noop=avg_noop,
            avg_accepts=avg_accepts,
            avg_length=eval_length,
            avg_convergence=avg_convergence,
        )
        self.db.update_eval(eval_id, stats)

        if convergence_results is not None:
            for agent, res in convergence_results.items():
                self.db.log_convergence(
                    eval_id,
                    agent_name=agent,
                    stats=dict(
                        convergence_pct=res["convergence_percentage"],
                        unique_actions=res["unique_actions"],
                        total_samples=res["total_samples"],
                    ),
                )
