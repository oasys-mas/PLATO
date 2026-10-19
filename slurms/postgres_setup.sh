#!/bin/bash
#SBATCH --time=7-00:00:00

#SBATCH --mem=32gb
#SBATCH --job-name=postgres_server
#SBATCH --error=postgres_server.err
#SBATCH --output=postgres_server.out
#SBATCH --dependency=singleton
#SBATCH --signal=B:SIGINT@60

# PostgreSQL server job for PLATO experiments
# Usage: sbatch postgres_setup.sh
# Creates environment file: DB/postgres_env.sh (source this in your jobs)
source ~/.bashrc
conda activate PLATO_ENV
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "🔧 Setting up PostgreSQL for PLATO experiments..."

# Step 1: Create directory structure and variables
export POSTGRES_HOME=$(pwd)/DB
mkdir -p $POSTGRES_HOME/{config,db/data,run,logs}

# Set proper permissions for PostgreSQL
chmod 755 $POSTGRES_HOME
chmod 700 $POSTGRES_HOME/db/data
chmod 755 $POSTGRES_HOME/run

# Generate password if it doesn't exist
if [ ! -f $POSTGRES_HOME/config/postgres-password ]; then
    uuidgen > $POSTGRES_HOME/config/postgres-password
    chmod 600 $POSTGRES_HOME/config/postgres-password
fi

export POSTGRES_PASSWORD_FILE=$POSTGRES_HOME/config/postgres-password
export POSTGRES_USER=$USER
export POSTGRES_DB=mydb
export PGDATA=$POSTGRES_HOME/db/data
export POSTGRES_HOST_AUTH_METHOD=md5
export POSTGRES_INITDB_ARGS="--data-checksums"
export POSTGRES_PORT=$(shuf -i 2000-65000 -n 1)

echo "Postgres server running on $(hostname) on port $POSTGRES_PORT"
echo "This job started at $(date +%Y-%m-%dT%T)"
echo "This job will end at $(squeue --noheader -j $SLURM_JOBID -o %e) (in $(squeue --noheader -j $SLURM_JOBID -o %L))"

# Write environment variables for other jobs to source
cat > $POSTGRES_HOME/postgres_env.sh << EOF
export POSTGRES_HOST=$(hostname)
export POSTGRES_PORT=$POSTGRES_PORT
export POSTGRES_USER=$USER
export POSTGRES_DB=mydb
EOF

echo "✅ Environment file created: $POSTGRES_HOME/postgres_env.sh"

module load apptainer
exec apptainer run \
    -B $POSTGRES_HOME/db/data:/var/lib/postgresql/data \
    -B $POSTGRES_HOME/run:/var/run/postgresql \
    -B $POSTGRES_HOME/logs:/var/log/postgresql \
    docker://postgres:11 \
    postgres -c "port=$POSTGRES_PORT" \
             -c "unix_socket_directories=/var/run/postgresql" \
             -c "logging_collector=on" \
             -c "log_directory=/var/log/postgresql" \
             -c "max_connections=1000" \
             -c "shared_buffers=1GB"