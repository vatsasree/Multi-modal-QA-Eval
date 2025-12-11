FROM nvcr.io/nvidia/pytorch:24.03-py3

# Define build arguments
ARG UID
ARG GID

# Ensure UID and GID are provided
RUN if [ -z "$UID" ] || [ -z "$GID" ]; then \
    echo "Error: UID and GID build arguments must be provided." >&2; \
    exit 1; \
    fi

# Set timezone and disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# 1) Install base packages (including keychain, tree, tmux, htop)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        poppler-utils \
        git \
        sudo \
        vim \
        screen \
        software-properties-common \
        tzdata \
        fonts-noto \
        keychain \
        tree \
        tmux \
        htop && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*  # Cleanup

    # 2) Set Python 3.11 as the default
    RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    update-alternatives --set python3 /usr/bin/python3.11
    
    ENV PATH="/usr/bin:$PATH"
    
    # Verify Python version as root (should print Python 3.11.x)
    RUN python3 --version
    
    # 3) Create a group and user, then create additional group 'vision' and add user to it
    RUN groupadd -f -g "${GID}" shanmukha_sreevatsa && \
    useradd    -u "${UID}" -g "${GID}" -m -s /bin/bash shanmukha_sreevatsa && \
    echo "shanmukha_sreevatsa:abcde1234" | chpasswd && \
    groupadd -f -g 1010 vision && \
    usermod -aG vision,sudo shanmukha_sreevatsa && \
    echo 'shanmukha_sreevatsa ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
    
    # Switch to the new user
    USER shanmukha_sreevatsa
    
    # Create and set the working directory
    WORKDIR /home/shanmukha_sreevatsa
    
    # Install nvitop via pip (as non-root user)
    RUN pip install --no-cache-dir nvitop
    
    # Install Miniconda
    RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
        bash miniconda.sh -b -p $HOME/miniconda && \
        rm miniconda.sh && \
        echo 'export PATH=$HOME/miniconda/bin:$PATH' >> ~/.bashrc
        
    # Set up keychain and tmux config for the user
RUN echo 'eval $(keychain --eval --agents ssh ~/.ssh/id_rsa)' >> ~/.bashrc && \
    echo 'set -g mouse on' > ~/.tmux.conf

# Verify Python version as the new user
RUN echo "Python version: $(python3 --version)"

CMD ["/bin/bash"]