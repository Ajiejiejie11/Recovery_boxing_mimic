if [ ! -v MAMBA_ROOT_PREFIX ]; then
    echo "WARNING: The MAMBA_ROOT_PREFIX environment variable is not set."
    echo "WARNING: This is required for mamba to work correctly as of 2.0."
    echo "WARNING: "
    echo "WARNING: For now, we are setting 'MAMBA_ROOT_PREFIX' to '/home/alan/Project/walking-boxing/magicbot-mimic/.miniforge3'."
    echo "WARNING: "
    echo "WARNING: Please make sure this is consistent with your installation or alternatively (by order of preference):"
    echo "WARNING:   - rerun 'mamba shell init' to initialize mamba for your current shell"
    echo "WARNING:   - manually set 'MAMBA_ROOT_PREFIX' to the root of your installation in your shell profile script."
    echo "WARNING:   - use the '-r,--root-prefix' CLI option when calling mamba."
    echo "WARNING: "
    echo "WARNING: This message originates from /home/alan/Project/walking-boxing/magicbot-mimic/.miniforge3/etc/profile.d/mamba.sh"
    export MAMBA_ROOT_PREFIX="/home/alan/Project/walking-boxing/magicbot-mimic/.miniforge3"
fi

__mamba_setup="$("/home/alan/Project/walking-boxing/magicbot-mimic/.miniforge3/bin/mamba" shell hook --shell posix 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias mamba="/home/alan/Project/walking-boxing/magicbot-mimic/.miniforge3/bin/mamba"  # Fallback on help from mamba activate
fi
unset __mamba_setup
