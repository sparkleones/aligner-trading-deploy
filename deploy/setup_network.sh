#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  AWS Network Setup Script                                                   ║
# ║  Configures Static IP (Elastic IP) binding for SEBI regulatory compliance   ║
# ║  Target: AWS EC2 M8g (Graviton4) in ap-south-1 (Mumbai)                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# SEBI 2026 Framework requires:
# 1. Static Public IP address registered with the broker
# 2. All API calls must originate from the registered IP
# 3. Dynamic residential IPs are NOT permitted for algo trading
#
# Usage:
#   chmod +x setup_network.sh
#   ./setup_network.sh [--allocate-eip] [--bind-eip <allocation-id>]
#
# Prerequisites:
#   - AWS CLI v2 configured with appropriate IAM permissions
#   - EC2 instance running in ap-south-1 (Mumbai) region
#   - IAM role with ec2:AllocateAddress, ec2:AssociateAddress permissions

set -euo pipefail

AWS_REGION="ap-south-1"
LOG_PREFIX="[NETWORK-SETUP]"

log_info() { echo "$LOG_PREFIX [INFO] $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ) $*"; }
log_warn() { echo "$LOG_PREFIX [WARN] $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ) $*"; }
log_error() { echo "$LOG_PREFIX [ERROR] $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ) $*" >&2; }

# ── Get Instance Metadata ───────────────────────────────────────────────────

get_instance_id() {
    # IMDSv2 (token-based)
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || echo "")

    if [ -n "$TOKEN" ]; then
        curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
            "http://169.254.169.254/latest/meta-data/instance-id" 2>/dev/null
    else
        curl -s "http://169.254.169.254/latest/meta-data/instance-id" 2>/dev/null
    fi
}

get_current_public_ip() {
    curl -s --connect-timeout 5 https://checkip.amazonaws.com/ 2>/dev/null || echo "unknown"
}

# ── Allocate Elastic IP ─────────────────────────────────────────────────────

allocate_elastic_ip() {
    log_info "Allocating new Elastic IP in $AWS_REGION..."

    RESULT=$(aws ec2 allocate-address \
        --domain vpc \
        --region "$AWS_REGION" \
        --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=algo-trading-static-ip},{Key=Purpose,Value=SEBI-Compliance}]" \
        --output json)

    ALLOCATION_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['AllocationId'])")
    PUBLIC_IP=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['PublicIp'])")

    log_info "Elastic IP allocated: $PUBLIC_IP (Allocation ID: $ALLOCATION_ID)"
    echo "$ALLOCATION_ID"
}

# ── Associate Elastic IP with Instance ───────────────────────────────────────

associate_elastic_ip() {
    local ALLOCATION_ID="$1"
    local INSTANCE_ID

    INSTANCE_ID=$(get_instance_id)
    if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "" ]; then
        log_error "Could not determine EC2 instance ID. Are you running on EC2?"
        exit 1
    fi

    log_info "Associating Elastic IP $ALLOCATION_ID with instance $INSTANCE_ID..."

    aws ec2 associate-address \
        --instance-id "$INSTANCE_ID" \
        --allocation-id "$ALLOCATION_ID" \
        --region "$AWS_REGION" \
        --allow-reassociation

    log_info "Elastic IP associated successfully"

    # Wait for IP to propagate
    sleep 5
    CURRENT_IP=$(get_current_public_ip)
    log_info "Current public IP: $CURRENT_IP"
}

# ── Configure Security Group ────────────────────────────────────────────────

configure_security_group() {
    local INSTANCE_ID
    INSTANCE_ID=$(get_instance_id)

    log_info "Configuring security group for algo trading..."

    # Get the security group ID attached to this instance
    SG_ID=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$AWS_REGION" \
        --query "Reservations[0].Instances[0].SecurityGroups[0].GroupId" \
        --output text)

    log_info "Security group: $SG_ID"

    # Allow outbound HTTPS to broker APIs
    aws ec2 authorize-security-group-egress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 \
        --region "$AWS_REGION" 2>/dev/null || log_info "HTTPS egress rule already exists"

    # Allow outbound WebSocket (typically port 443)
    # Allow outbound gRPC (internal only, already covered by VPC)

    # Restrict inbound to SSH only from admin IP
    log_info "Security group configured for trading operations"
}

# ── Verify Network Configuration ────────────────────────────────────────────

verify_network() {
    log_info "═══════════════════════════════════════════════════════"
    log_info "  Network Configuration Verification"
    log_info "═══════════════════════════════════════════════════════"

    # Check public IP
    PUBLIC_IP=$(get_current_public_ip)
    log_info "Public IP:     $PUBLIC_IP"

    # Check region
    log_info "AWS Region:    $AWS_REGION"

    # Check instance type
    INSTANCE_ID=$(get_instance_id)
    if [ -n "$INSTANCE_ID" ]; then
        INSTANCE_TYPE=$(aws ec2 describe-instances \
            --instance-ids "$INSTANCE_ID" \
            --region "$AWS_REGION" \
            --query "Reservations[0].Instances[0].InstanceType" \
            --output text 2>/dev/null || echo "unknown")
        log_info "Instance:      $INSTANCE_ID ($INSTANCE_TYPE)"
    fi

    # Test broker API connectivity
    log_info "Testing broker API connectivity..."
    KITE_PING=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
        "https://api.kite.trade/instruments" 2>/dev/null || echo "000")
    log_info "Kite API:      HTTP $KITE_PING"

    FYERS_PING=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
        "https://api-t1.fyers.in/api/v3" 2>/dev/null || echo "000")
    log_info "Fyers API:     HTTP $FYERS_PING"

    # Measure latency to NSE/broker endpoints
    log_info "Measuring latency to broker endpoints..."
    LATENCY=$(curl -s -o /dev/null -w "%{time_total}" --connect-timeout 5 \
        "https://api.kite.trade" 2>/dev/null || echo "timeout")
    log_info "Kite latency:  ${LATENCY}s"

    log_info "═══════════════════════════════════════════════════════"

    # Write config for the application
    cat > /app/.network_config <<NETCFG
STATIC_IP=$PUBLIC_IP
AWS_REGION=$AWS_REGION
INSTANCE_ID=${INSTANCE_ID:-local}
VERIFIED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
NETCFG

    log_info "Network config written to /app/.network_config"
    log_info ""
    log_info "IMPORTANT: Register this Static IP ($PUBLIC_IP) with your broker"
    log_info "for SEBI algo trading compliance."
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    log_info "Indian Options Trading System — Network Setup"
    log_info "Target region: $AWS_REGION (Mumbai)"

    case "${1:-verify}" in
        --allocate-eip)
            ALLOC_ID=$(allocate_elastic_ip)
            associate_elastic_ip "$ALLOC_ID"
            configure_security_group
            verify_network
            ;;
        --bind-eip)
            if [ -z "${2:-}" ]; then
                log_error "Usage: $0 --bind-eip <allocation-id>"
                exit 1
            fi
            associate_elastic_ip "$2"
            configure_security_group
            verify_network
            ;;
        verify|--verify)
            verify_network
            ;;
        *)
            echo "Usage: $0 [--allocate-eip | --bind-eip <allocation-id> | --verify]"
            echo ""
            echo "  --allocate-eip         Allocate a new Elastic IP and bind it"
            echo "  --bind-eip <id>        Bind an existing Elastic IP allocation"
            echo "  --verify               Verify current network configuration (default)"
            exit 0
            ;;
    esac
}

main "$@"
