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

function practiceProgressMutationArn(stack: Stack, endpoint: string): string {
  let apiId: string;
  try {
    apiId = new URL(endpoint).hostname.split('.')[0] ?? '';
  } catch {
    throw new Error('APPSYNC_GRAPHQL_ENDPOINT must be a valid AppSync endpoint');
  }
  if (!/^[a-z0-9]+$/.test(apiId)) {
    throw new Error('APPSYNC_GRAPHQL_ENDPOINT must identify an AppSync API');
  }
  return stack.formatArn({
    service: 'appsync',
    resource: `apis/${apiId}/types/Mutation/fields/updatePracticeGenerationProgress`,
  });
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
  /** Explicit non-secret environment values supplied by the deployment target. */
  runtimeEnvironment?: Readonly<Record<string, string>>;
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

    const { spec, deploymentEnvironment, mcpSpec, credentials, runtimeEnvironment = {} } = props;
    const practiceEnabled = runtimeEnvironment.PRACTICE_GENERATION_ENABLED === 'true';
    const patternIntelligenceEnabled = runtimeEnvironment.PATTERN_INTELLIGENCE_ENABLED === 'true';
    const patternReuseEnabled = runtimeEnvironment.PATTERN_INTELLIGENCE_REUSE_ENABLED === 'true';
    // Case-folded to match the Python runtime, which reads this same value with
    // .lower() == "true".  A bare === 'true' would let "TRUE" enable promotion in
    // the runtime while leaving it without the dynamodb:PutItem grant.
    const questionBankPromotionEnabled =
      runtimeEnvironment.PRACTICE_QUESTION_BANK_PROMOTION_ENABLED?.toLowerCase() === 'true';
    // Phase D: "shadow" validates without serving, "on" serves. Both need discovery.
    const questionSemanticMode = (
      runtimeEnvironment.PRACTICE_QUESTION_SEMANTIC_REUSE_MODE ?? 'off'
    ).toLowerCase();
    const questionSemanticEnabled =
      questionSemanticMode === 'shadow' || questionSemanticMode === 'on';

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
    const examProfileTableArn = StringParameter.valueForStringParameter(
      this,
      '/meritranker/agent-runtime/v1/exam-profile/table-arn'
    );
    const practiceAssessmentTableArn = StringParameter.valueForStringParameter(
      this,
      '/meritranker/agent-runtime/v1/practice/mock-test-quiz/table-arn'
    );
    const practiceQuestionTableArn = StringParameter.valueForStringParameter(
      this,
      '/meritranker/agent-runtime/v1/practice/question/table-arn'
    );
    const practiceQuestionBankTableArn = StringParameter.valueForStringParameter(
      this,
      '/meritranker/agent-runtime/v1/practice/question-bank/table-arn'
    );
    const practiceQuestionTestIndex = StringParameter.valueForStringParameter(
      this,
      '/meritranker/agent-runtime/v1/practice/question/test-id-index-name'
    );
    const practiceQuestionBankCategoryIndex = StringParameter.valueForStringParameter(
      this,
      '/meritranker/agent-runtime/v1/practice/question-bank/category-index-name'
    );
    const practiceQuestionBankReuseIndex = StringParameter.valueForStringParameter(
      this,
      '/meritranker/agent-runtime/v1/practice/question-bank/reuse-index-name'
    );
    const patternTableName = patternIntelligenceEnabled
      ? StringParameter.valueForStringParameter(
          this,
          '/meritranker/agent-runtime/v1/pattern-intelligence/pattern/table-name'
        )
      : '';
    const patternTableArn = patternIntelligenceEnabled
      ? StringParameter.valueForStringParameter(
          this,
          '/meritranker/agent-runtime/v1/pattern-intelligence/pattern/table-arn'
        )
      : '';
    const questionVectorIndexArn = questionSemanticEnabled
      ? StringParameter.valueForStringParameter(
          this,
          '/meritranker/agent-runtime/v1/practice/question-bank/vector/index-arn'
        )
      : '';
    const patternVectorIndexArn = patternIntelligenceEnabled
      ? StringParameter.valueForStringParameter(
          this,
          '/meritranker/agent-runtime/v1/pattern-intelligence/vector/index-arn'
        )
      : '';
    const practiceQuestionBankPatternIndex = patternIntelligenceEnabled
      ? StringParameter.valueForStringParameter(
          this,
          '/meritranker/agent-runtime/v1/practice/question-bank/pattern-index-name'
        )
      : '';
    const practiceAttemptTableName = patternReuseEnabled
      ? StringParameter.valueForStringParameter(this, '/meritranker/agent-runtime/v1/practice/attempt/table-name')
      : '';
    const practiceAttemptTableArn = patternReuseEnabled
      ? StringParameter.valueForStringParameter(this, '/meritranker/agent-runtime/v1/practice/attempt/table-arn')
      : '';
    const practiceAttemptUserActivityIndex = patternReuseEnabled
      ? StringParameter.valueForStringParameter(
          this,
          '/meritranker/agent-runtime/v1/practice/attempt/user-activity-index-name'
        )
      : '';
    // Student runtime credits. Gated so a deployment with enforcement disabled
    // never requires the credit parameters to exist.
    const studentCreditEnforcementEnabled =
      runtimeEnvironment.STUDENT_CREDIT_ENFORCEMENT_ENABLED?.toLowerCase() === 'true';
    const userCreditsTableName = studentCreditEnforcementEnabled
      ? StringParameter.valueForStringParameter(this, '/meritranker/agent-runtime/v1/credits/user-credits/table-name')
      : '';
    const userCreditsTableArn = studentCreditEnforcementEnabled
      ? StringParameter.valueForStringParameter(this, '/meritranker/agent-runtime/v1/credits/user-credits/table-arn')
      : '';
    const creditLedgerTableName = studentCreditEnforcementEnabled
      ? StringParameter.valueForStringParameter(this, '/meritranker/agent-runtime/v1/credits/credit-ledger/table-name')
      : '';
    const creditLedgerTableArn = studentCreditEnforcementEnabled
      ? StringParameter.valueForStringParameter(this, '/meritranker/agent-runtime/v1/credits/credit-ledger/table-arn')
      : '';
    const ssmParameterArns = [
      'conversation-history/table-name',
      'conversation-history/table-arn',
      'conversation-session/table-name',
      'conversation-session/table-arn',
      'exam-profile/table-name',
      'exam-profile/table-arn',
      'practice/resource-contract-version',
      'practice/reuse-key-contract-version',
      'practice/mock-test-quiz/table-name',
      'practice/mock-test-quiz/table-arn',
      'practice/question/table-name',
      'practice/question/table-arn',
      'practice/question/test-id-index-name',
      'practice/question-bank/table-name',
      'practice/question-bank/table-arn',
      'practice/question-bank/category-index-name',
      'practice/question-bank/reuse-index-name',
      ...(patternIntelligenceEnabled
        ? [
            'practice/question-bank/pattern-index-name',
            'pattern-intelligence/pattern/table-name',
            'pattern-intelligence/pattern/table-arn',
            'pattern-intelligence/vector/index-arn',
          ]
        : []),
      ...(patternReuseEnabled
        ? ['practice/attempt/table-name', 'practice/attempt/table-arn', 'practice/attempt/user-activity-index-name']
        : []),
    ].map(resourceName =>
      this.formatArn({
        service: 'ssm',
        resource: 'parameter',
        resourceName: `meritranker/agent-runtime/v1/${resourceName}`,
      })
    );
    const practiceEndpoint = runtimeEnvironment.APPSYNC_GRAPHQL_ENDPOINT;
    if (practiceEnabled && !practiceEndpoint) {
      throw new Error('APPSYNC_GRAPHQL_ENDPOINT is required when practice generation is enabled');
    }
    for (const environment of this.application.environments.values()) {
      const runtimeResource = environment.runtime.node.findChild('Resource');
      if (!(runtimeResource instanceof aws_bedrockagentcore.CfnRuntime)) {
        throw new Error('AgentCore Runtime resource is unavailable');
      }
      for (const [name, value] of Object.entries(runtimeEnvironment)) {
        runtimeResource.addPropertyOverride(`EnvironmentVariables.${name}`, value);
      }
      if (patternIntelligenceEnabled) {
        runtimeResource.addPropertyOverride('EnvironmentVariables.DYNAMODB_PATTERN_TABLE', patternTableName);
        runtimeResource.addPropertyOverride('EnvironmentVariables.S3_VECTOR_PATTERN_INDEX_ARN', patternVectorIndexArn);
        runtimeResource.addPropertyOverride('EnvironmentVariables.S3_VECTOR_PATTERN_INDEX_NAME', 'patterns-v1');
        runtimeResource.addPropertyOverride(
          'EnvironmentVariables.DYNAMODB_QUESTION_BANK_PATTERN_INDEX',
          practiceQuestionBankPatternIndex
        );
        runtimeResource.addPropertyOverride(
          'EnvironmentVariables.DYNAMODB_QUESTION_BANK_TABLE',
          StringParameter.valueForStringParameter(
            this,
            '/meritranker/agent-runtime/v1/practice/question-bank/table-name'
          )
        );
      }
      if (patternReuseEnabled) {
        runtimeResource.addPropertyOverride(
          'EnvironmentVariables.DYNAMODB_PRACTICE_ATTEMPT_TABLE',
          practiceAttemptTableName
        );
        runtimeResource.addPropertyOverride(
          'EnvironmentVariables.DYNAMODB_PRACTICE_ATTEMPT_USER_INDEX',
          practiceAttemptUserActivityIndex
        );
      }
      if (studentCreditEnforcementEnabled) {
        runtimeResource.addPropertyOverride(
          'EnvironmentVariables.DYNAMODB_USER_CREDITS_TABLE',
          userCreditsTableName
        );
        runtimeResource.addPropertyOverride(
          'EnvironmentVariables.DYNAMODB_CREDIT_LEDGER_TABLE',
          creditLedgerTableName
        );
      }
      removeDefaultMemoryGrant(environment.runtime.role);
      for (const memory of this.application.memories.values()) {
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['bedrock-agentcore:CreateEvent', 'bedrock-agentcore:ListEvents'],
            resources: [memory.memoryArn],
          })
        );
      }
      // QuestionBank promotion is trusted-Question capability, not Pattern reuse, so
      // its write grant is gated on the promotion flag alone.  It also sits outside
      // the memory loop above: nesting it there made the grant depend on how many
      // memories exist.
      if (questionSemanticEnabled) {
        runtimeResource.addPropertyOverride(
          'EnvironmentVariables.S3_VECTOR_QUESTION_INDEX_ARN',
          questionVectorIndexArn
        );
        // Discovery only, questions-v1 only. Nothing on patterns-v1, no write actions.
        // GetVectors accompanies QueryVectors because S3 Vectors requires it as a
        // dependent permission when a query uses metadata filters and returnMetadata;
        // application code never calls get_vectors, DynamoDB remains authoritative.
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['s3vectors:QueryVectors', 's3vectors:GetVectors'],
            resources: [questionVectorIndexArn],
          })
        );
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['bedrock:InvokeModel'],
            resources: [
              this.formatArn({
                service: 'bedrock',
                account: '',
                resource: 'foundation-model',
                resourceName: 'amazon.titan-embed-text-v2:0',
              }),
            ],
          })
        );
      }
      if (questionBankPromotionEnabled) {
        // PutItem creates the reusable row; UpdateItem exists solely so an existing
        // Question can acquire authoritative Pattern linkage it did not have when it
        // was first promoted.  Nothing else may mutate QuestionBank.
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['dynamodb:PutItem', 'dynamodb:UpdateItem'],
            resources: [practiceQuestionBankTableArn],
          })
        );
      }
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['ssm:GetParameters'],
          resources: ssmParameterArns,
        })
      );
      if (patternIntelligenceEnabled) {
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['s3vectors:QueryVectors'],
            resources: [patternVectorIndexArn],
          })
        );
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['bedrock:InvokeModel'],
            resources: [
              this.formatArn({
                service: 'bedrock',
                account: '',
                resource: 'foundation-model',
                resourceName: 'amazon.titan-embed-text-v2:0',
              }),
            ],
          })
        );
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['dynamodb:BatchGetItem'],
            resources: [patternTableArn],
          })
        );
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['dynamodb:Query'],
            resources: [`${practiceQuestionBankTableArn}/index/${practiceQuestionBankPatternIndex}`],
          })
        );
      }
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:Scan'],
          resources: [examProfileTableArn],
        })
      );
      if (studentCreditEnforcementEnabled) {
        // GetItem reads the authoritative balance for admission and recovers an
        // existing settlement. TransactWriteItems performs the only write: the
        // conditional ledger insert plus wallet decrement, together or not at
        // all. The runtime never creates a wallet, so no PutItem is granted,
        // and the non-transactional UpdateItem path is deliberately excluded.
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['dynamodb:GetItem', 'dynamodb:TransactWriteItems'],
            resources: [userCreditsTableArn, creditLedgerTableArn],
          })
        );
      }
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:GetItem', 'dynamodb:PutItem', 'dynamodb:Query'],
          resources: [historyTableArn, `${historyTableArn}/index/ConversationHistoryByConversation`],
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:BatchGetItem', 'dynamodb:GetItem', 'dynamodb:PutItem', 'dynamodb:UpdateItem'],
          resources: [sessionTableArn],
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:DescribeTable'],
          resources: [practiceAssessmentTableArn, practiceQuestionTableArn, practiceQuestionBankTableArn],
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:GetItem', 'dynamodb:PutItem', 'dynamodb:UpdateItem'],
          resources: [practiceAssessmentTableArn],
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:BatchGetItem', 'dynamodb:PutItem', 'dynamodb:UpdateItem'],
          resources: [practiceQuestionTableArn],
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:DeleteItem', 'dynamodb:TransactWriteItems'],
          resources: [practiceAssessmentTableArn, practiceQuestionTableArn],
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:Query'],
          resources: [`${practiceQuestionTableArn}/index/${practiceQuestionTestIndex}`],
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:BatchGetItem'],
          resources: [practiceQuestionBankTableArn],
        })
      );
      environment.runtime.addToPolicy(
        new iam.PolicyStatement({
          actions: ['dynamodb:Query'],
          resources: [
            `${practiceQuestionBankTableArn}/index/${practiceQuestionBankCategoryIndex}`,
            `${practiceQuestionBankTableArn}/index/${practiceQuestionBankReuseIndex}`,
          ],
        })
      );
      if (practiceEnabled && practiceEndpoint) {
        environment.runtime.addToPolicy(
          new iam.PolicyStatement({
            actions: ['appsync:GraphQL'],
            resources: [practiceProgressMutationArn(this, practiceEndpoint)],
          })
        );
        if (patternReuseEnabled) {
          environment.runtime.addToPolicy(
            new iam.PolicyStatement({
              actions: ['dynamodb:Query'],
              resources: [`${practiceAttemptTableArn}/index/${practiceAttemptUserActivityIndex}`],
            })
          );
        }
      }
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
