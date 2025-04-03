import os
import subprocess
import sys
import socket

def get_ip():
    """Get the local IP address to use for the certificate"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def generate_certificates():
    """Generate self-signed SSL certificates for development"""
    print("Generating self-signed certificates for development HTTPS...")
    
    # Create certs directory if it doesn't exist
    cert_dir = "certs"
    if not os.path.exists(cert_dir):
        os.makedirs(cert_dir)
    
    # Get paths for cert and key files
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")
    
    # Get the IP address for the SAN
    ip_address = get_ip()
    
    # Create OpenSSL configuration with Subject Alternative Name (SAN)
    openssl_config = f"""[req]
    distinguished_name = req_distinguished_name
    req_extensions = v3_req
    prompt = no

    [req_distinguished_name]
    C = US
    ST = State
    L = City
    O = Organization
    OU = OrganizationalUnit
    CN = localhost

    [v3_req]
    keyUsage = critical, digitalSignature, keyAgreement
    extendedKeyUsage = serverAuth
    subjectAltName = @alt_names

    [alt_names]
    DNS.1 = localhost
    IP.1 = 127.0.0.1
    IP.2 = {ip_address}
    """
    
    # Write config to file
    config_file = os.path.join(cert_dir, "openssl.cnf")
    with open(config_file, 'w') as f:
        f.write(openssl_config)
    
    # Generate certificates using OpenSSL
    try:
        subprocess.run([
            'openssl', 'req', '-x509', '-newkey', 'rsa:4096', 
            '-keyout', key_file, '-out', cert_file,
            '-days', '365', '-nodes', 
            '-config', config_file
        ], check=True)
        
        print(f"\nSuccess! Self-signed certificates generated in {cert_dir}/")
        print(f"Certificate file: {cert_file}")
        print(f"Key file: {key_file}")
        print(f"\nYour server can now be accessed via HTTPS at:")
        print(f"https://localhost:8080")
        print(f"https://{ip_address}:8080")
        print("\nNote: Browsers will show a security warning because it's self-signed.")
        print("You'll need to click 'Advanced' and 'Proceed' to access the site.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating certificates: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: OpenSSL is not installed or not in the PATH.")
        print("Please install OpenSSL and try again.")
        sys.exit(1)

if __name__ == "__main__":
    generate_certificates() 