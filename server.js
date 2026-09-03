import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import Groq from 'groq-sdk';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

// Load environment variables
dotenv.config();

// Disable SSL strict rejection
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

const app = express();
const PORT = process.env.PORT || 3001;

// ============================================
// 🔥 CONFIGURAÇÕES
// ============================================
const GROQ_API_KEY = process.env.GROQ_API_KEY;
if (!GROQ_API_KEY) {
    console.warn('⚠️ GROQ_API_KEY não configurada. Use variáveis de ambiente.');
}
const MODEL = process.env.MODEL || 'openai/gpt-oss-120b';
const TEMPERATURE = parseFloat(process.env.TEMPERATURE || '0.7') || 0.7;
const MAX_TOKENS = parseInt(process.env.MAX_TOKENS || '2048', 10) || 2048;

const FALLBACK_MODELS = ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'];

// ============================================
// 🔥 BASE DE CONHECIMENTO - ESTÉTICA FEMININA
// ============================================
const CONHECIMENTO_ESTETICA = {
    pele: {
        rotina: [
            'Limpeza facial (manhã e noite)',
            'Tonificação',
            'Hidratação',
            'Proteção solar (manhã)',
            'Esfoliação (1-2x por semana)',
            'Máscaras faciais (1-2x por semana)'
        ],
        dicas: [
            'Use água morna para lavar o rosto, nunca água quente',
            'Aplique protetor solar mesmo em dias nublados',
            'Durma com o rosto limpo',
            'Troque a fronha do travesseiro regularmente'
        ],
        tipos: {
            'normal': 'Pele equilibrada, pouca oleosidade, poros finos',
            'seca': 'Pele áspera, descamação, sensação de repuxamento',
            'oleosa': 'Brilho excessivo, poros dilatados, acne',
            'mista': 'Oleosa na zona T, seca nas bochechas',
            'sensível': 'Vermelhidão, coceira, reação a produtos'
        }
    },
    makeup: {
        passos: [
            'Preparação da pele (limpeza, hidratação, primer)',
            'Base ou BB Cream',
            'Corretivo',
            'Pó compacto ou solto',
            'Contorno e iluminador',
            'Blush',
            'Sombra e delineador',
            'Máscara de cílios',
            'Lábios (batom, gloss)',
            'Fixador (spray)'
        ],
        dicas: [
            'Sempre use primer para aumentar a duração da maquiagem',
            'Aplique base com movimentos ascendentes',
            'Use pincéis limpos para evitar bactérias',
            'Remova toda a maquiagem antes de dormir'
        ]
    },
    cabelos: {
        tipos: {
            'liso': 'Fácil de pentear, brilho natural',
            'ondulado': 'Forma ondas naturais, volume médio',
            'cacheado': 'Cachos definidos, volume',
            'crespo': 'Fios muito enrolados, volume intenso'
        },
        cuidados: [
            'Lavar com shampoo adequado ao seu tipo de cabelo',
            'Usar condicionador apenas no comprimento',
            'Hidratação semanal',
            'Corte regular (a cada 2-3 meses)',
            'Evitar água muito quente'
        ]
    },
    alimentacao: {
        alimentos: [
            'Água (hidratação é essencial)',
            'Frutas vermelhas (antioxidantes)',
            'Vegetais verdes (vitaminas)',
            'Peixes ricos em ômega 3',
            'Castanhas e nozes',
            'Chá verde (antioxidante)'
        ],
        dicas: [
            'Beba pelo menos 2L de água por dia',
            'Evite açúcar refinado',
            'Consuma alimentos ricos em colágeno',
            'Reduza o consumo de álcool e cafeína'
        ]
    },
    bem_estar: {
        beneficios: [
            'Melhora a circulação sanguínea',
            'Reduz o estresse',
            'Aumenta a produção de colágeno',
            'Melhora a qualidade do sono',
            'Aumenta a autoestima'
        ],
        sugeridos: [
            'Caminhada diária (30 min)',
            'Ioga ou pilates',
            'Treino de força',
            'Dança',
            'Meditação'
        ]
    }
};

// ============================================
// 🔥 MIDDLEWARES - CORRIGIDO
// ============================================

// ✅ CORS atualizado para aceitar todas as origens (produção)
app.use(cors({
    origin: '*', // Permite todas as origens para teste
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allowedHeaders: ['Content-Type', 'Authorization', 'Accept', 'X-Requested-With']
}));

// ✅ Helmet com configurações flexíveis
app.use(helmet({
    crossOriginResourcePolicy: { policy: "cross-origin" },
    contentSecurityPolicy: false,
    crossOriginEmbedderPolicy: false,
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: 60 * 1000,
    max: 100, // Aumentado para evitar bloqueios em testes
    message: { error: 'Muitas requisições. Aguarde um momento.' },
    standardHeaders: true,
    legacyHeaders: false,
});
app.use('/api', limiter);

app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

// Logger
app.use((req, _res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${req.method} ${req.path}`);
    next();
});

// ============================================
// 🔥 CLIENTE GROQ
// ============================================
let groqClient = null;

function getGroqClient() {
    if (!GROQ_API_KEY) {
        console.warn('⚠️ GROQ_API_KEY não configurada');
        return null;
    }
    if (!groqClient) {
        try {
            groqClient = new Groq({
                apiKey: GROQ_API_KEY,
                timeout: 30000,
                maxRetries: 2,
            });
            console.log('✅ Cliente Groq inicializado');
        } catch (err) {
            console.error('❌ Erro ao inicializar Groq:', err);
            return null;
        }
    }
    return groqClient;
}

async function callGroqWithFallback(groq, messages, temperature = TEMPERATURE, maxTokens = MAX_TOKENS) {
    const modelsToTry = [MODEL, ...FALLBACK_MODELS.filter((m) => m !== MODEL)];

    let lastError = null;
    for (const modelToUse of modelsToTry) {
        try {
            console.log(`📤 Tentando modelo: ${modelToUse}`);
            const resp = await groq.chat.completions.create({
                model: modelToUse,
                messages,
                temperature,
                max_tokens: maxTokens,
                top_p: 0.9,
            });
            console.log(`✅ Sucesso com modelo: ${modelToUse}`);
            return { completion: resp, modelUsed: modelToUse };
        } catch (err) {
            lastError = err;
            console.warn(`❌ Modelo ${modelToUse} falhou: ${err.message}`);
        }
    }
    throw lastError || new Error('Todos os modelos falharam');
}

// ============================================
// 🔥 SISTEMA DE TEMAS E CATEGORIAS
// ============================================

const TEMAS = {
    'pele': ['pele', 'facial', 'acne', 'oleosidade', 'ressecamento', 'manchas', 'rugas', 'poros', 'limpeza facial'],
    'makeup': ['maquiagem', 'makeup', 'base', 'corretivo', 'pó', 'blush', 'sombra', 'delineador', 'batom', 'gloss'],
    'cabelos': ['cabelo', 'cabelos', 'cacho', 'crespo', 'liso', 'ondulado', 'capilar', 'hidratação capilar', 'queda', 'caspa'],
    'alimentacao': ['alimentação', 'comida', 'dieta', 'nutrição', 'vitaminas', 'antioxidantes', 'colágeno', 'água'],
    'bem_estar': ['bem estar', 'bem-estar', 'exercício', 'ioga', 'pilates', 'meditação', 'estresse', 'saúde mental', 'autoestima'],
    'make': ['make', 'maquiagem', 'makeup'],
    'cuidados': ['cuidado', 'rotina', 'dicas', 'tratamento']
};

function detectarTema(pergunta) {
    const perguntaLower = pergunta.toLowerCase();
    for (const [tema, palavras] of Object.entries(TEMAS)) {
        for (const palavra of palavras) {
            if (perguntaLower.includes(palavra)) {
                return tema;
            }
        }
    }
    return null;
}

function buscarConhecimentoLocal(tema, pergunta) {
    const perguntaLower = pergunta.toLowerCase();
    let resposta = [];

    if (tema === 'pele') {
        if (perguntaLower.includes('rotina') || perguntaLower.includes('como cuidar')) {
            resposta.push('📋 **Rotina de Cuidados com a Pele:**');
            CONHECIMENTO_ESTETICA.pele.rotina.forEach((item, i) => {
                resposta.push(`${i + 1}. ${item}`);
            });
        }
        if (perguntaLower.includes('dica')) {
            resposta.push('💡 **Dicas para uma Pele Saudável:**');
            CONHECIMENTO_ESTETICA.pele.dicas.forEach((dica) => {
                resposta.push(`• ${dica}`);
            });
        }
        if (perguntaLower.includes('tipo')) {
            resposta.push('📊 **Tipos de Pele:**');
            for (const [tipo, desc] of Object.entries(CONHECIMENTO_ESTETICA.pele.tipos)) {
                resposta.push(`• **${tipo}**: ${desc}`);
            }
        }
    }

    if (tema === 'makeup' || tema === 'make') {
        if (perguntaLower.includes('passo') || perguntaLower.includes('como fazer') || perguntaLower.includes('aplicar')) {
            resposta.push('💄 **Passos para uma Maquiagem Perfeita:**');
            CONHECIMENTO_ESTETICA.makeup.passos.forEach((item, i) => {
                resposta.push(`${i + 1}. ${item}`);
            });
        }
        if (perguntaLower.includes('dica')) {
            resposta.push('💡 **Dicas de Maquiagem:**');
            CONHECIMENTO_ESTETICA.makeup.dicas.forEach((dica) => {
                resposta.push(`• ${dica}`);
            });
        }
    }

    if (tema === 'cabelos') {
        if (perguntaLower.includes('tipo')) {
            resposta.push('💇‍♀️ **Tipos de Cabelo:**');
            for (const [tipo, desc] of Object.entries(CONHECIMENTO_ESTETICA.cabelos.tipos)) {
                resposta.push(`• **${tipo}**: ${desc}`);
            }
        }
        if (perguntaLower.includes('cuidado') || perguntaLower.includes('tratamento')) {
            resposta.push('🧴 **Cuidados com os Cabelos:**');
            CONHECIMENTO_ESTETICA.cabelos.cuidados.forEach((item) => {
                resposta.push(`• ${item}`);
            });
        }
    }

    if (tema === 'alimentacao') {
        resposta.push('🥗 **Alimentos que Favorecem a Beleza:**');
        CONHECIMENTO_ESTETICA.alimentacao.alimentos.forEach((item) => {
            resposta.push(`• ${item}`);
        });
        resposta.push('\n💡 **Dicas de Alimentação:**');
        CONHECIMENTO_ESTETICA.alimentacao.dicas.forEach((dica) => {
            resposta.push(`• ${dica}`);
        });
    }

    if (tema === 'bem_estar') {
        resposta.push('🧘‍♀️ **Benefícios dos Exercícios para a Beleza:**');
        CONHECIMENTO_ESTETICA.bem_estar.beneficios.forEach((item) => {
            resposta.push(`• ${item}`);
        });
        resposta.push('\n💪 **Atividades Sugeridas:**');
        CONHECIMENTO_ESTETICA.bem_estar.sugeridos.forEach((item) => {
            resposta.push(`• ${item}`);
        });
    }

    return resposta.length > 0 ? resposta.join('\n') : null;
}

// ============================================
// 🔥 ROTAS DA API
// ============================================

// Health check
app.get('/health', (_req, res) => {
    res.json({
        status: 'online',
        service: 'Bella - Assistente de Estética Feminina',
        version: '1.0.0',
        timestamp: new Date().toISOString(),
        config: {
            model: MODEL,
            temperature: TEMPERATURE,
            hasApiKey: !!GROQ_API_KEY,
        },
        temas: Object.keys(TEMAS),
    });
});

// Status do bot
app.get('/api/bot/status', (_req, res) => {
    res.json({
        active: true,
        name: 'Bella - Assistente de Estética Feminina',
        version: '1.0.0',
        model: MODEL,
        hasGroqKey: !!GROQ_API_KEY,
        mode: 'estetica-feminina',
        temas: Object.keys(TEMAS),
        disponivel: true,
    });
});

// Temas disponíveis
app.get('/api/bot/temas', (_req, res) => {
    res.json({
        temas: [
            { id: 'pele', nome: 'Cuidados com a Pele', icone: '✨' },
            { id: 'makeup', nome: 'Maquiagem', icone: '💄' },
            { id: 'cabelos', nome: 'Cuidados com os Cabelos', icone: '💇‍♀️' },
            { id: 'alimentacao', nome: 'Alimentação e Beleza', icone: '🥗' },
            { id: 'bem_estar', nome: 'Bem-estar e Exercícios', icone: '🧘‍♀️' },
        ]
    });
});

// ============================================
// 🔥 CHAT PRINCIPAL - CORRIGIDO
// ============================================

app.post('/api/bot/chat', async (req, res) => {
    try {
        const { message, history = [] } = req.body;

        console.log('📥 Body recebido:', { message, historyLength: history.length });

        if (!message || typeof message !== 'string' || message.trim().length === 0) {
            return res.status(400).json({ error: 'A mensagem não pode estar vazia.' });
        }

        const userQuery = message.trim();
        console.log(`\n💬 Bella recebeu: "${userQuery}"`);

        const tema = detectarTema(userQuery);
        console.log(`📊 Tema detectado: ${tema || 'geral'}`);

        let conhecimentoLocal = null;
        if (tema) {
            conhecimentoLocal = buscarConhecimentoLocal(tema, userQuery);
            if (conhecimentoLocal) {
                console.log('📚 Resposta do conhecimento local encontrada');
            }
        }

        // ✅ CONSTRUÇÃO DO HISTÓRICO MELHORADA
        const systemPrompt = `Você é a Bella, uma assistente virtual especializada em estética feminina, beleza, cuidados pessoais e bem-estar.

🧠 **PERSONALIDADE:**
- Amigável, acolhedora, empática e encorajadora
- Especialista em estética feminina
- Fala sobre beleza, cuidados com a pele, maquiagem, cabelos, alimentação saudável e bem-estar
- Responde SEMPRE em português do Brasil
- Usa emojis e linguagem calorosa 💖

📚 **ÁREAS DE CONHECIMENTO:**
- 🧖‍♀️ Cuidados com a Pele (rotinas, tipos de pele, tratamentos)
- 💄 Maquiagem (técnicas, produtos, dicas)
- 💇‍♀️ Cuidados com os Cabelos (tipos, tratamentos, hidratação)
- 🥗 Alimentação e Beleza (nutrição, vitaminas, colágeno)
- 🧘‍♀️ Bem-estar e Exercícios (saúde mental, autoestima)

${conhecimentoLocal ? `\n📚 **INFORMAÇÃO ESPECÍFICA PARA ESTA PERGUNTA:**\n${conhecimentoLocal}\n` : ''}

⚠️ **REGRAS:**
1. Responda SEMPRE em português do Brasil
2. Seja acolhedora e empática
3. Dê dicas práticas e aplicáveis
4. Se não souber algo, diga honestamente e sugira onde procurar
5. Incentive a autoestima e o autocuidado
6. Use emojis para tornar a conversa mais agradável

🎯 **OBJETIVO:** Ser a melhor amiga de beleza e bem-estar da usuária!`;

        // ✅ CONSTRUÇÃO DO HISTÓRICO MELHORADA
        const messagesToSend = [
            { role: 'system', content: systemPrompt },
        ];

        // Adiciona histórico se existir
        if (Array.isArray(history) && history.length > 0) {
            // Pega apenas as últimas 10 mensagens para não sobrecarregar
            const limitedHistory = history.slice(-10);
            for (const h of limitedHistory) {
                if (h && (h.role === 'user' || h.role === 'assistant') && typeof h.content === 'string') {
                    messagesToSend.push({ role: h.role, content: h.content });
                }
            }
        }

        // ✅ SEMPRE adiciona a mensagem atual
        messagesToSend.push({ role: 'user', content: userQuery });

        console.log(`📨 Total de mensagens para IA: ${messagesToSend.length}`);

        // 🔥 USA IA GROQ
        const groq = getGroqClient();
        
        if (groq) {
            try {
                console.log(`📤 Enviando para Groq (${MODEL})...`);

                const { completion, modelUsed } = await callGroqWithFallback(groq, messagesToSend, TEMPERATURE, MAX_TOKENS);
                let reply = completion.choices[0]?.message?.content || 'Desculpe, não consegui processar sua pergunta.';

                if (conhecimentoLocal && !reply.includes(conhecimentoLocal.substring(0, 50))) {
                    reply += `\n\n💖 **Dica Bella:**\n${conhecimentoLocal}`;
                }

                console.log(`🤖 Resposta gerada (${reply.length} caracteres)`);

                return res.json({
                    reply,
                    mode: 'groq-ai',
                    model: modelUsed,
                    tema: tema || 'geral',
                    tokens: completion.usage?.total_tokens || 0,
                    hasLocalKnowledge: !!conhecimentoLocal,
                });

            } catch (error) {
                console.error('❌ Erro na IA:', error);
                if (conhecimentoLocal) {
                    return res.json({
                        reply: `💖 **Bella aqui!** \n\n${conhecimentoLocal}\n\n✨ *Usei meu conhecimento interno para te responder. Se quiser mais detalhes, me pergunte novamente!*`,
                        mode: 'conhecimento-local',
                        tema: tema || 'geral',
                        hasLocalKnowledge: true,
                    });
                }
                return res.json({
                    reply: `💖 **Olá! Eu sou a Bella, sua assistente de estética feminina!**\n\n` +
                           `✨ **Posso ajudar com:**\n` +
                           `• 🧖‍♀️ Cuidados com a pele\n` +
                           `• 💄 Maquiagem e makes\n` +
                           `• 💇‍♀️ Cuidados com os cabelos\n` +
                           `• 🥗 Alimentação saudável\n` +
                           `• 🧘‍♀️ Bem-estar e autoestima\n\n` +
                           `💬 **Me pergunte algo específico!**\n` +
                           `Exemplo: *"Como cuidar da pele oleosa?"*`,
                    mode: 'fallback',
                    tema: tema || 'geral',
                });
            }
        }

        if (conhecimentoLocal) {
            return res.json({
                reply: `💖 **Bella aqui!** \n\n${conhecimentoLocal}\n\n✨ *Posso te ajudar com mais dúvidas sobre estética e beleza!*`,
                mode: 'conhecimento-local',
                tema: tema || 'geral',
                hasLocalKnowledge: true,
            });
        }

        return res.json({
            reply: `💖 **Olá! Eu sou a Bella!**\n\n` +
                   `✨ **Posso te ajudar com:**\n` +
                   `• 🧖‍♀️ Rotina de cuidados com a pele\n` +
                   `• 💄 Dicas de maquiagem\n` +
                   `• 💇‍♀️ Tratamentos capilares\n` +
                   `• 🥗 Alimentação para beleza\n` +
                   `• 🧘‍♀️ Bem-estar e autoestima\n\n` +
                   `💬 **Me pergunte algo específico!**\n` +
                   `Exemplo: *"Qual a melhor rotina para pele seca?"*`,
            mode: 'inicial',
            tema: 'geral',
        });

    } catch (error) {
        console.error('❌ Erro no chat:', error);
        return res.status(500).json({
            error: error.message,
            reply: `💖 **Ops! Ocorreu um erro.**\n\n` +
                   `Detalhe: ${error.message || 'Erro desconhecido'}\n\n` +
                   `💬 **Tente perguntar de outra forma:**\n` +
                   `• "Como cuidar da pele oleosa?"\n` +
                   `• "Dicas de maquiagem para iniciantes"`,
            mode: 'error',
        });
    }
});

// ============================================
// 🔥 ROTA DE TESTE
// ============================================

app.get('/api/test', (_req, res) => {
    res.json({
        message: 'Bella está online! 💖',
        status: 'ok',
        version: '1.0.0',
        endpoints: {
            health: '/health',
            chat: '/api/bot/chat',
            temas: '/api/bot/temas',
            status: '/api/bot/status',
        },
        exemplo: {
            method: 'POST',
            url: '/api/bot/chat',
            body: { message: 'Como cuidar da pele oleosa?' }
        }
    });
});

// ============================================
// 🔥 404 - Rota não encontrada
// ============================================
app.use('*', (req, res) => {
    res.status(404).json({
        error: 'Rota não encontrada',
        availableRoutes: [
            'GET  /health',
            'GET  /api/bot/status',
            'GET  /api/bot/temas',
            'POST /api/bot/chat',
            'GET  /api/test',
        ]
    });
});

// ============================================
// 🔥 INICIALIZAÇÃO
// ============================================

app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n${'='.repeat(70)}`);
    console.log(`💖 BELLA - ASSISTENTE DE ESTÉTICA FEMININA`);
    console.log(`${'='.repeat(70)}`);
    console.log(`🚀 Servidor: http://0.0.0.0:${PORT}`);
    console.log(`📦 Modelo: ${MODEL}`);
    console.log(`🔑 Groq: ${GROQ_API_KEY ? '✅ Configurada' : '❌ Não configurada'}`);
    console.log(`📚 Temas disponíveis: ${Object.keys(TEMAS).join(', ')}`);
    console.log(`${'='.repeat(70)}`);
    console.log(`\n💬 EXEMPLOS DE PERGUNTAS:`);
    console.log(`  🧖‍♀️ "Como cuidar da pele oleosa?"`);
    console.log(`  💄 "Dicas de maquiagem para iniciantes"`);
    console.log(`  💇‍♀️ "Como hidratar cabelos cacheados?"`);
    console.log(`  🥗 "O que comer para ter uma pele bonita?"`);
    console.log(`  🧘‍♀️ "Exercícios para melhorar a autoestima"`);
    console.log(`${'='.repeat(70)}\n`);
});

// Tratamento de encerramento
process.on('SIGINT', () => {
    console.log('\n💖 Bella foi encerrada. Até logo!');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n💖 Bella foi encerrada. Até logo!');
    process.exit(0);
});

process.on('uncaughtException', (error) => {
    console.error('❌ Erro não capturado:', error);
});

process.on('unhandledRejection', (reason) => {
    console.error('❌ Rejeição não tratada:', reason);
});