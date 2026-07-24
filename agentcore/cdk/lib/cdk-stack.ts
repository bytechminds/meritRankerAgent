import {
  AgentCoreApplication,
  AgentCoreMcp,
  type AgentCoreProjectSpec,
  type AgentCoreMcpSpec,
} from '@aws/agentcore-cdk';
import { CfnOutput, Stack, aws_bedrockagentcore, type StackProps } from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { StringParameter } from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';

const AGENTCORE_L3_MEMORY_ACTIONS = new Set([
  'bedrock-agentcore:ListMemoryRecords',
  'bedrock-agentcore:RetrieveMemoryRecords',
  'bedrock-agentcore:GetEvent',
  'bedrock-agentcore:GetMemory',
  'bedrock-agentcore:GetMemoryRecord',
  'bedrock-agentcore:ListActors',
  'bedrock-agentcore:ListEvents',
  'bedrock-agentcore:ListSessions',
  'bedrock-agentcore:CreateEvent',
  'bedrock-agentcore:DeleteEvent',
  'bedrock-agentcore:DeleteMemoryRecord',
]);

function removeDefaultMemoryGrant(role: iam.IRole): void {
  for (const child of role.node.findAll()) {
    if (!(child instanceof iam.Policy)) continue;
    // CDK exposes additions but not removals; the assertion test locks this L3 override.
    const document = child.document as unknown as {
      statements: iam.PolicyStatement[];
    };
    for (let index = document.statements.length - 1; index >= 0; index -= 1) {
      const raw = document.statements[index].toStatementJson().Action;
      const actions = Array.isArray(raw) ? raw : [raw];
      if (actions.some(action => AGENTCORE_L3_MEMORY_ACTIONS.has(String(action)))) {
        document.statements.splice(index, 1);
      }
    }
  }
}

export interface AgentCoreStackProps extends StackProps {
  /**
   * The AgentCore project specification containing agents, memories, and credentials.
   */
  spec: AgentCoreProjectSpec;
  /**
   * Deployment target used to isolate environment-specific resource names.
   */
  deploymentEnvironment: string;
  /**
   * The MCP specification containing gateways and servers.
   */
  mcpSpec?: AgentCoreMcpSpec;
  /**
   * Credential provider ARNs from deployed state, keyed by credential name.
   */
  credentials?: Record<string, { credentialProviderArn: string; clientSecretArn?: string }>;
}

/**
 * CDK Stack that deploys AgentCore infrastructure.
 *
 * This is a thin wrapper that instantiates L3 constructs.
 * All resource logic and outputs are contained within the L3 constructs.
 */
export class AgentCoreStack extends Stack {
  /** The AgentCore application containing all agent environments */
  public readonly application: AgentCoreApplication;

  constructor(scope: Construct, id: string, props: AgentCoreStackProps) {
    super(scope, id, props);

    const { spec, deploymentEnvironment, mcpSpec, credentials } = props;

    // Create AgentCoreApplication with all agents
    this.application = new AgentCoreApplication(this, 'Application', {
      spec,
    });

    const environmentSuffix = deploymentEnvironment.toLowerCase().replace(/[^a-z0-9_]/g, '_');
    for (const [memoryName, memory] of this.application.memories) {
      const physicalName = `${memoryName}_${environmentSuffix}`;
      if (physicalName.length > 48) {
        throw new Error(`Environment-scoped AgentCore Memory name exceeds 48 characters: ${physicalName}`);
      }
      const resource = memory.node.findChild('Resource');
      if (!(resource instanceof aws_bedrockagentcore.CfnMemory)) {
        throw new Error(`AgentCore Memory resource is unavailable for ${memoryName}`);
      }
      resource.name = physicalName;
    }

    const historyTableArn = StringParameter.valueForStringParameter(
      this,
      '/meritranker/agent-runtime/v1/conversation-history/table-arn'
    );
    const sessionTableArn = StringParameter.valueForStringParameter(
      this,
      '/meritranker/agent-runtime/v1/conversation-session/table-arn'
    );
    const ssmParameterArns = [
      'conversation-history/table-name',
      'conversation-history/table-arn',
      'conversation-session/table-name',
      'conversation-session/table-arn',
    ].map(resourceName =>
      this.formatArn({
        service: 'ssm',
        resource: 'parameter',
        resourceName: `meritranker/agent-runtime/v1/${resourceName}`,
      })
    );
    for (const environment of this.application.environments.values()) {
      removeDefaultMemoryGrant(environment.runtime.role);
      for (const memory of this.application.memories.values()) {
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['bedrock-agentcore:CreateEvent', 'bedrock-agentcore:ListEvents'],
            resources: [memory.memoryArn],
          })
        );
      }
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['ssm:GetParameters'],
          resources: ssmParameterArns,
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:GetItem', 'dynamodb:PutItem', 'dynamodb:Query'],
          resources: [historyTableArn, `${historyTableArn}/index/ConversationHistoryByConversation`],
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:GetItem', 'dynamodb:PutItem', 'dynamodb:UpdateItem'],
          resources: [sessionTableArn],
        })
      );
    }

    // Create AgentCoreMcp if there are gateways configured
    if (mcpSpec?.agentCoreGateways && mcpSpec.agentCoreGateways.length > 0) {
      new AgentCoreMcp(this, 'Mcp', {
        projectName: spec.name,
        mcpSpec,
        agentCoreApplication: this.application,
        credentials,
        projectTags: spec.tags,
      });
    }

    // Stack-level output
    new CfnOutput(this, 'StackNameOutput', {
      description: 'Name of the CloudFormation Stack',
      value: this.stackName,
    });
  }
}
