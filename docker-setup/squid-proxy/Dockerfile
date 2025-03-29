# Use a lightweight base image
FROM ubuntu:24.04

# Install Squid and other necessary tools
RUN apt-get update && apt-get install -y squid curl

# Copy the Squid configuration file
COPY ./squid.conf /etc/squid/squid.conf

# Expose the Squid proxy port
EXPOSE 3128

# Start Squid service
CMD ["squid", "-N"]