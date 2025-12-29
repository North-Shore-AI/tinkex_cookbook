import Config

# Disable CrucibleFramework Repo (we don't use Ecto persistence)
config :crucible_framework, enable_repo: false

# Import environment specific config
import_config "#{config_env()}.exs"
