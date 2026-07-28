#!/usr/bin/env bash

export TIMESTAMP=$(date +%s)
export LOG_DIR="${HOME}/.devcontainer/logs/${TIMESTAMP}"

function chown_dir() {
    # ディレクトリの所有者を変更する関数
    # Args:
    #   $1: ディレクトリのパス
    local dir=$1
    if [ -d "$dir" ]; then
        sudo chown -R $(whoami):$(whoami) "$dir"
    fi
}

function setup_python_project() {
    # Python プロジェクトインストール関数
    # Args:
    #   $1: プロジェクトディレクトリのパス
    local project_dir=$1
    local log_file="${LOG_DIR}/${project_dir}.log"
    chown_dir "${project_dir}/.venv"
    mkdir -p "${LOG_DIR}"

    pushd "$project_dir"                >> "${log_file}"
    if [ -f "poetry.lock" ]; then
        # Poetry プロジェクト
        poetry install                  >> "${log_file}"
        printf "\e[36m- Completed to setup the poetry project.: \e[0m\e[36m${project_dir}\e[0m\n"
    elif [ -f "pyproject.toml" ]; then
        # uv プロジェクト
        uv python install 3.14          >> "${log_file}"
        uv sync --dev                   >> "${log_file}"
        printf "\e[36m- Completed to setup the uv(Python) project.: \e[0m\e[36m${project_dir}\e[0m\n"
    elif [ -f "requirements.txt" ]; then
        # pip プロジェクト
        pip install -r requirements.txt >> "${log_file}"
        printf "\e[36m- Completed to setup the pip project.: \e[0m\e[36m${project_dir}\e[0m\n"
    else
        printf "\e[33m- Skipped to setup Python project...: \e[0m\e[36m${project_dir}\e[0m\n"
    fi
    popd
}

function setup_nodejs_project() {
    # Node.js プロジェクトインストール関数
    # Args:
    #   $1: プロジェクトディレクトリのパス
    local project_dir=$1
    local log_file="${LOG_DIR}/${project_dir}.log"
    chown_dir "${project_dir}/node_modules"
    mkdir -p "${LOG_DIR}"

    pushd "$project_dir" >> "${log_file}"
    if [ -f "pnpm-lock.yaml" ]; then
        # pnpm プロジェクト
        pnpm install     >> "${log_file}"
        printf "\e[36m- Completed to setup the pnpm project.: \e[0m\e[36m${project_dir}\e[0m\n"
    elif [ -f "yarn.lock" ]; then
        # yarn プロジェクト
        yarn install     >> "${log_file}"
        printf "\e[36m- Completed to setup the yarn project.: \e[0m\e[36m${project_dir}\e[0m\n"
    elif [ -f "package-lock.json" ]; then
        # npm プロジェクト
        npm install      >> "${log_file}"
        printf "\e[36m- Completed to setup the npm project.: \e[0m\e[36m${project_dir}\e[0m\n"
    else
        printf "\e[33m- Skipped to setup Node.js project...: \e[0m\e[36m${project_dir}\e[0m\n"
    fi
    popd                 >> "${log_file}"
}

function setup_serena() {
    # Serena MCP サーバ セットアップ関数
    chown_dir .serena

    if [ ! -d ".serena/cache" ]; then
        uvx --no-env-file --from git+https://github.com/oraios/serena serena project index
        printf "\e[36m- Completed to setup Serena MCP server.\e[0m\n"
    fi
}

function main() {
    # メイン関数
    printf "\e[34mpostCreateCommand\e[0m\n"

    local script_start=$(date +%s%3N)

    chown_dir "/usr/local/share/nvm"
    chown_dir ".pnpm-store"
    chown_dir "/home/vscode/.cache"
    chown_dir "/home/vscode/.cache/uv"
    chown_dir ".serena/cache"

    chown_dir "agent/.venv"
    chown_dir "cdk/node_modules"
    chown_dir "lambdas/node_modules"
    chown_dir "web/node_modules"
    chown_dir "web/styled-system"

    # 隠しディレクトリ以外のディレクトリを探索
    for dir in */; do
        # 末尾のスラッシュを削除
        dir="${dir%/}"

        # 隠しディレクトリをスキップ
        if [[ "$dir" == .* ]]; then
            continue
        fi

        # package.jsonが存在する場合、Node.jsプロジェクトとしてセットアップ
        if [ -f "$dir/package.json" ]; then
            setup_nodejs_project "$dir" &
        fi

        # pyproject.tomlが存在する場合、Pythonプロジェクトとしてセットアップ
        if [ -f "$dir/pyproject.toml" ]; then
            setup_python_project "$dir" &
        fi
    done

    # プロジェクトのセットアップとSerenaとpre-commitのセットアップを並列実行
    setup_serena &

    # すべてのセットアップが完了するまで待機
    wait

    local script_end=$(date +%s%3N)
    local total_duration=$((script_end - script_start))
    local seconds=$((total_duration / 1000))
    local milliseconds=$((total_duration % 1000))
    printf "\e[32mSetup complete! Total time: %d.%03d [s]\e[0m\n" $seconds $milliseconds
}

main
