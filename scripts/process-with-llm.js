import { pipeline, env } from '@xenova/transformers';
import { promises as fs } from 'fs';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configure transformers for GitHub Actions
env.allowRemoteFiles = true;
env.allowLocalFiles = true;
env.cacheDir = path.join(__dirname, '../.cache');

// Suppress ONNX runtime warnings - more aggressive approach
process.env.ORT_LOG_LEVEL = '3'; // Only show errors
process.env.ONNX_DISABLE_WARNINGS = '1';
process.env.ONNXRUNTIME_LOG_LEVEL = '3'; // ERROR level only
process.env.OMP_NUM_THREADS = '1'; // Reduce threading warnings
process.env.ONNX_LOGGING_LEVEL = '3'; // ERROR level
process.env.ONNXRUNTIME_LOG_SEVERITY_LEVEL = '3'; // ERROR level

// Suppress Node.js warnings
process.removeAllListeners('warning');
process.on('warning', () => {}); // Suppress warnings

// Suppress specific console warnings from ONNX runtime
const originalConsoleWarn = console.warn;
console.warn = function(...args) {
  const message = args.join(' ');
  // Skip ONNX runtime warnings about removing unused initializers
  if (message.includes('CleanUnusedInitializersAndNodeArgs') || 
      message.includes('Removing initializer') ||
      message.includes('onnxruntime') ||
      message.includes('should be removed from the model') ||
      message.includes('[W:onnxruntime')) {
    return;
  }
  originalConsoleWarn.apply(console, args);
};

// Also suppress stderr warnings from child processes
const originalStderrWrite = process.stderr.write;
process.stderr.write = function(chunk, encoding, fd) {
  if (typeof chunk === 'string' && 
      (chunk.includes('CleanUnusedInitializersAndNodeArgs') ||
       chunk.includes('Removing initializer') ||
       chunk.includes('onnxruntime') ||
       chunk.includes('should be removed from the model') ||
       chunk.includes('[W:onnxruntime'))) {
    return;
  }
  return originalStderrWrite.call(process.stderr, chunk, encoding, fd);
};

// Categories for MTM digest (Maharashtra, MSME, jobs, policy)
const categories = [
  'maharashtra-state-local',
  'msme-industry-trade',
  'government-schemes',
  'jobs-recruitment',
  'education-skills',
  'tax-compliance-gst',
  'banking-credit',
  'policy-regulation',
  'legal-judiciary',
  'civic-public-service'
];

let classifier, summarizer, ner;

// Initialize pipelines with working models only
async function initializeModels() {
  try {
    console.log('Loading classifier...');
    classifier = await pipeline('zero-shot-classification', 'Xenova/distilbert-base-uncased-mnli', {
      cache_dir: env.cacheDir,
      quantized: true
    });
    console.log('✓ Classifier loaded');

    console.log('Loading summarizer...');
    summarizer = await pipeline('summarization', 'Xenova/distilbart-cnn-6-6', {
      cache_dir: env.cacheDir,
      quantized: true
    });
    console.log('✓ Summarizer loaded');

    console.log('Loading NER...');
    ner = await pipeline('ner', 'Xenova/bert-base-NER', {
      cache_dir: env.cacheDir,
      quantized: true
    });
    console.log('✓ NER loaded');

    return true;
  } catch (error) {
    console.log('❌ Failed to load models:', error.message);
    console.log('🔄 Using rule-based processing...');
    return false;
  }
}

// Rule-based category classification as fallback
function classifyCategory(title, source) {
  const titleLower = title.toLowerCase();
  const sourceLower = source.toLowerCase();
  
  if (titleLower.match(/\b(high court|supreme court|ordinance|petition|verdict|bail)\b/) ||
      sourceLower.includes('prs')) {
    return { category: 'legal-judiciary', confidence: 0.88 };
  }
  
  if (titleLower.match(/\b(recruitment|vacancy|ssc|upsc|exam date|admit card|notification)\b/) ||
      sourceLower.includes('jobs')) {
    return { category: 'jobs-recruitment', confidence: 0.88 };
  }
  
  if (titleLower.match(/\b(admission|scholarship|skill|university|board exam|ugc|education)\b/) ||
      sourceLower.includes('education')) {
    return { category: 'education-skills', confidence: 0.85 };
  }
  
  if (titleLower.match(/\b(maharashtra|mumbai|pune|thane|nagpur|bmc|mantralaya|shiv sena|ncp)\b/) ||
      sourceLower.includes('mumbai') || sourceLower.includes('pune') || sourceLower.includes('state_mh')) {
    return { category: 'maharashtra-state-local', confidence: 0.85 };
  }
  
  if (titleLower.match(/\b(gst|income tax|audit|compliance|itr|duty)\b/)) {
    return { category: 'tax-compliance-gst', confidence: 0.85 };
  }
  
  if (titleLower.match(/\b(rbi|repo rate|credit|npas|bank loan|nbfc|interest rate)\b/) ||
      sourceLower.includes('rbi')) {
    return { category: 'banking-credit', confidence: 0.85 };
  }
  
  if (titleLower.match(/\b(msme|udyam|startup|sme|industry|manufacturing|export|midc)\b/) ||
      sourceLower.includes('companies')) {
    return { category: 'msme-industry-trade', confidence: 0.82 };
  }
  
  if (titleLower.match(/\b(scheme|subsidy|pm-?kisan|ayushman|mgnrega|welfare|grant)\b/)) {
    return { category: 'government-schemes', confidence: 0.84 };
  }
  
  if (titleLower.match(/\b(power outage|water supply|disaster|relief|health alert|weather warning)\b/)) {
    return { category: 'civic-public-service', confidence: 0.78 };
  }
  
  if (titleLower.match(/\b(cabinet|ministry|policy|parliament|bill |ordinance|notification|circular)\b/) ||
      sourceLower.includes('pib')) {
    return { category: 'policy-regulation', confidence: 0.8 };
  }
  
  return { category: 'policy-regulation', confidence: 0.55 };
}

function extractEntities(title) {
  const entities = [];
  const orgs = [
    'RBI', 'SEBI', 'GST Council', 'Ministry of Finance', 'Ministry of MSME',
    'Maharashtra', 'Mumbai', 'Pune', 'Nagpur', 'State Bank of India', 'SBI',
    'NABARD', 'SIDBI', 'UIDAI', 'EPFO', 'ESIC', 'NHAI', 'NHM', 'PIB'
  ];
  orgs.forEach(org => {
    if (title.toLowerCase().includes(org.toLowerCase())) {
      entities.push({ text: org, label: 'ORG' });
    }
  });
  const schemes = ['GST', 'Udyam', 'MSME', 'PM-KISAN', 'Ayushman', 'MUDRA', 'Stand Up India'];
  schemes.forEach(s => {
    if (title.toLowerCase().includes(s.toLowerCase())) {
      entities.push({ text: s, label: 'MISC' });
    }
  });
  const tags = ['recruitment', 'tender', 'policy', 'budget', 'court', 'exam'];
  tags.forEach(tag => {
    if (title.toLowerCase().includes(tag)) {
      entities.push({ text: tag, label: 'TECH' });
    }
  });
  return entities;
}

// Reading density: plain news vs legal or fiscal jargon
function calculateDifficulty(title, entities) {
  const plain = ['inaugurates', 'announces', 'launches', 'flags off', 'visits', 'meets'];
  const legal = ['ordinance', 'verdict', 'tribunal', 'petition', 'bail', 'section ', 'act ', 'code '];
  const fiscal = ['repo rate', 'liquidity', 'fiscal deficit', 'securitisation', 'impairment', 'circular'];
  
  let difficulty = 4;
  const titleLower = title.toLowerCase();
  
  plain.forEach(term => {
    if (titleLower.includes(term)) difficulty -= 1;
  });
  legal.forEach(term => {
    if (titleLower.includes(term)) difficulty += 2;
  });
  fiscal.forEach(term => {
    if (titleLower.includes(term)) difficulty += 2;
  });
  
  difficulty += Math.min(entities.length, 4) * 0.5;
  
  return Math.min(Math.max(Math.round(difficulty), 1), 10);
}

// Generate summary
async function generateSummary(title, metaDescription, source, useAI = false) {
  // Check if metaDescription is meaningful (not just Reddit boilerplate)
  const isRedditBoilerplate = metaDescription && (
    metaDescription.includes('submitted by') && metaDescription.includes('[link]') ||
    metaDescription.trim().length < 30 ||
    metaDescription.includes('https://preview.redd.it') ||
    metaDescription.match(/^https?:\/\//)
  );
  
  // Use meaningful metaDescription for summarization, skip Reddit boilerplate
  const contentToSummarize = metaDescription && metaDescription.trim() && !isRedditBoilerplate && metaDescription.length > 50 
    ? metaDescription.trim() 
    : null;
  
  if (!contentToSummarize && source && source.toLowerCase().includes('reddit')) {
    return createTopicBasedSummary(title);
  }
  
  if (!useAI || !summarizer || !contentToSummarize) {
    // Rule-based summary with meaningful content
    if (contentToSummarize) {
      const cleaned = contentToSummarize.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
      return cleaned.length > 120 ? cleaned.substring(0, 120) + '...' : cleaned;
    }
    return createTopicBasedSummary(title);
  }
  
  try {
    const summary = await summarizer(contentToSummarize, {
      max_length: 60,
      min_length: 25
    });
    return summary[0].summary_text;
  } catch (error) {
    // Fallback to rule-based summary on error
    if (contentToSummarize) {
      const cleaned = contentToSummarize.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
      return cleaned.length > 120 ? cleaned.substring(0, 120) + '...' : cleaned;
    }
    return createTopicBasedSummary(title);
  }
}

function createTopicBasedSummary(title) {
  const titleLower = title.toLowerCase();
  if (titleLower.match(/\b(recruitment|vacancy|exam|ssc|upsc|admit)\b/)) {
    return 'Update on jobs, recruitment, or competitive exams relevant to readers.';
  }
  if (titleLower.match(/\b(scheme|subsidy|welfare|grant|ministry)\b/)) {
    return 'Government scheme, welfare, or policy announcement.';
  }
  if (titleLower.match(/\b(rbi|repo|gst|tax|bank|credit)\b/)) {
    return 'Banking, tax, or regulatory update for businesses and households.';
  }
  if (titleLower.match(/\b(maharashtra|mumbai|pune|thane)\b/)) {
    return 'Maharashtra state or city-level update.';
  }
  if (titleLower.match(/\b(court|ordinance|bill|parliament)\b/)) {
    return 'Legal or legislative development.';
  }
  return 'Public interest update for the MTM community digest.';
}

// Main processing function
async function processArticlesWithAI() {
  console.log('🤖 Starting article processing...');
  
  // Load raw articles
  const rawDataPath = path.join(__dirname, '../data/latest-raw.json');
  const rawData = JSON.parse(await fs.readFile(rawDataPath, 'utf-8'));
  
  // Load existing processed articles to avoid reprocessing
  const processedDataPath = path.join(__dirname, '../data/latest-processed.json');
  let existingProcessed = { articles: [] };
  
  try {
    const existingData = await fs.readFile(processedDataPath, 'utf-8');
    existingProcessed = JSON.parse(existingData);
    console.log(`📋 Found ${existingProcessed.articles.length} already processed articles`);
  } catch (error) {
    console.log('📋 No existing processed data found, processing all articles');
  }
  
  // Load rejected articles cache to avoid reprocessing low-confidence articles
  const rejectedDataPath = path.join(__dirname, '../data/rejected-articles.json');
  let rejectedArticles = { articles: [] };
  
  try {
    const rejectedData = await fs.readFile(rejectedDataPath, 'utf-8');
    rejectedArticles = JSON.parse(rejectedData);
    console.log(`🚫 Found ${rejectedArticles.articles.length} previously rejected articles`);
  } catch (error) {
    console.log('🚫 No rejected articles cache found, starting fresh');
  }
  
  // Create sets for quick lookup
  const processedIds = new Set(existingProcessed.articles.map(a => a.id));
  const rejectedIds = new Set(rejectedArticles.articles.map(a => a.id));
  
  // Filter out already processed AND rejected articles
  const articlesToProcess = rawData.articles.filter(article => 
    !processedIds.has(article.id) && !rejectedIds.has(article.id)
  );
  
  // Apply configurable processing limit for testing (via env var)
  const testLimit = process.env.PROCESSING_LIMIT ? parseInt(process.env.PROCESSING_LIMIT) : null;
  const finalArticlesToProcess = testLimit ? articlesToProcess.slice(0, testLimit) : articlesToProcess;
  
  console.log(`📊 Found ${rawData.articles.length} total articles, ${articlesToProcess.length} new articles to process`);
  if (rejectedArticles.articles.length > 0) {
    console.log(`⏭️ Skipping ${rejectedArticles.articles.length} previously rejected articles`);
  }
  if (testLimit && testLimit < articlesToProcess.length) {
    console.log(`🧪 TESTING MODE: Processing only first ${testLimit} articles (set PROCESSING_LIMIT=${testLimit})`);
  }
  
  if (finalArticlesToProcess.length === 0) {
    console.log('✅ All articles already processed! Updating metadata...');
    
    // Apply 15-day rolling cleanup to existing articles too
    const cleanupThresholdDays = 15;
    const cleanupThreshold = new Date(Date.now() - cleanupThresholdDays * 24 * 60 * 60 * 1000);
    
    const beforeCleanup = existingProcessed.articles.length;
    const recentArticles = existingProcessed.articles.filter(article => {
      const pubDate = new Date(article.pubDate || article.published_at);
      return pubDate > cleanupThreshold;
    });
    
    const cleanedUpCount = beforeCleanup - recentArticles.length;
    if (cleanedUpCount > 0) {
      console.log(`🧹 Cleaned up ${cleanedUpCount} articles older than ${cleanupThresholdDays} days from existing data`);
    }
    
    // Update the latest file with recent articles only (15-day rolling window)
    const latestData = {
      ...rawData,
      articles: recentArticles,
      processedAt: new Date().toISOString(),
      processingMethod: 'cached',
      totalArticles: recentArticles.length,
      cleanupApplied: cleanedUpCount > 0,
      cleanedUpCount: cleanedUpCount,
      rollingWindowDays: cleanupThresholdDays
    };
    
    // Create today's file with ONLY today's articles
    const today = new Date().toISOString().split('T')[0]; // YYYY-MM-DD format
    const todayStart = new Date(today + 'T00:00:00.000Z');
    const todayEnd = new Date(today + 'T23:59:59.999Z');
    
    // Filter articles that were published or processed today
    const todaysArticles = recentArticles.filter(article => {
      const pubDate = new Date(article.pubDate || article.published_at);
      const processedDate = new Date(article.processed_at || article.crawledAt);
      
      // Include articles published today OR processed today
      return (pubDate >= todayStart && pubDate <= todayEnd) || 
             (processedDate >= todayStart && processedDate <= todayEnd);
    });
    
    const dailyData = {
      ...rawData,
      articles: todaysArticles,
      processedAt: new Date().toISOString(),
      processingMethod: 'cached',
      totalArticles: todaysArticles.length,
      dailyArticlesCount: todaysArticles.length,
      filterDate: today
    };
    
    // Save files
    const latestPath = path.join(__dirname, '../data/latest-processed.json');
    const datePath = path.join(__dirname, `../data/${today}-processed.json`);
    
    await fs.writeFile(latestPath, JSON.stringify(latestData, null, 2));
    await fs.writeFile(datePath, JSON.stringify(dailyData, null, 2));
    
    console.log(`💾 Updated: ${latestPath} (${recentArticles.length} articles in 15-day rolling window)`);
    console.log(`📅 Historical backup: ${datePath} (${todaysArticles.length} today's articles)`);
    console.log('🎉 Processing completed (no new articles)!');
    return latestData;
  }
  
  // Initialize AI models only if we have articles to process
  const useAI = await initializeModels();
  
  console.log(`🧠 Processing ${finalArticlesToProcess.length} new articles with ${useAI ? 'LLM' : 'rule-based'} digest tagging...`);
  
  const newlyProcessedArticles = [];
  const newlyRejectedArticles = [];
  const categoryStats = {};
  let duplicateCount = 0;
  let rejectedLowConfidence = 0;
  
  for (let i = 0; i < finalArticlesToProcess.length; i++) {
    const article = finalArticlesToProcess[i];
    console.log(`Processing ${i+1}/${finalArticlesToProcess.length}: ${article.title.substring(0, 50)}...`);
    
    let result;
    
    if (useAI && classifier) {
      try {
        // AI-powered classification
        const classification = await classifier(article.title, categories);
        const topLabel = classification.labels[0];
        const confidence = classification.scores[0];
        
        result = {
          category: topLabel,
          confidence: confidence
        };
      } catch (error) {
        // Fallback to rule-based
        result = classifyCategory(article.title, article.source);
      }
    } else {
      // Rule-based classification
      result = classifyCategory(article.title, article.source);
    }
    
    // Extract entities
    let entities = [];
    if (useAI && ner) {
      try {
        const nerResults = await ner(article.title);
        entities = nerResults.filter(entity => entity.score > 0.8);
      } catch (error) {
        entities = extractEntities(article.title);
      }
    } else {
      entities = extractEntities(article.title);
    }
    
    // Calculate difficulty
    const difficulty = calculateDifficulty(article.title, entities);
    
    // Generate summary
    const summary = await generateSummary(article.title, article.metaDescription, article.source, useAI);
    
    // Apply confidence threshold filter - reject articles below threshold (configurable via PROCESS_CONFIDENCE_THRESHOLD env var)
    const confidenceThreshold = parseFloat(process.env.PROCESS_CONFIDENCE_THRESHOLD || '0.25');
    if (result.confidence < confidenceThreshold) {
      console.log(`❌ Rejected low confidence (${(result.confidence * 100).toFixed(1)}%): ${article.title.substring(0, 60)}...`);
      
      // Save rejected article to cache to avoid reprocessing
      const rejectedArticle = {
        ...article,
        rejectedReason: 'low_confidence',
        confidence: result.confidence,
        confidenceThreshold: confidenceThreshold,
        rejected_at: new Date().toISOString()
      };
      
      newlyRejectedArticles.push(rejectedArticle);
      rejectedLowConfidence++;
      continue; // Skip this article
    }
    
    // Create processed article
    const processedArticle = {
      ...article,
      category: result.category,
      confidence: result.confidence,
      difficulty: difficulty,
      entities: entities,
      summary: summary,
      language: 'en', // Default to English since we removed language detection
      processed_at: new Date().toISOString()
    };
    
    newlyProcessedArticles.push(processedArticle);
    
    // Update stats
    categoryStats[result.category] = (categoryStats[result.category] || 0) + 1;
  }
  
  // Merge newly processed articles with existing ones
  const allProcessedArticles = [...existingProcessed.articles, ...newlyProcessedArticles];
  
  // Sort by publication date (newest first)
  allProcessedArticles.sort((a, b) => new Date(b.pubDate || b.published_at) - new Date(a.pubDate || a.published_at));
  
  // Apply 15-day rolling cleanup to latest-processed.json to maintain consistency
  const cleanupThresholdDays = 15;
  const cleanupThreshold = new Date(Date.now() - cleanupThresholdDays * 24 * 60 * 60 * 1000);
  
  const beforeCleanup = allProcessedArticles.length;
  const recentArticles = allProcessedArticles.filter(article => {
    const pubDate = new Date(article.pubDate || article.published_at);
    return pubDate > cleanupThreshold;
  });
  
  const cleanedUpCount = beforeCleanup - recentArticles.length;
  if (cleanedUpCount > 0) {
    console.log(`🧹 Cleaned up ${cleanedUpCount} articles older than ${cleanupThresholdDays} days from latest-processed.json`);
  }
  
  // Create data for latest file (contains recent articles only - 15-day rolling window)
  const latestData = {
    ...rawData,
    articles: recentArticles,
    processedAt: new Date().toISOString(),
    totalArticles: recentArticles.length,
    categories: [...new Set(recentArticles.map(a => a.category))],
    processingMethod: useAI ? 'ai-powered' : 'rule-based',
    newArticlesProcessed: newlyProcessedArticles.length,
    existingArticlesKept: existingProcessed.articles.length,
    cleanupApplied: cleanedUpCount > 0,
    cleanedUpCount: cleanedUpCount,
    rollingWindowDays: cleanupThresholdDays
  };
  
  // Create data for daily file (contains ONLY today's articles)
  const today = new Date().toISOString().split('T')[0]; // YYYY-MM-DD format
  const todayStart = new Date(today + 'T00:00:00.000Z');
  const todayEnd = new Date(today + 'T23:59:59.999Z');
  
  // Filter articles that were published or processed today
  const todaysArticles = recentArticles.filter(article => {
    const pubDate = new Date(article.pubDate || article.published_at);
    const processedDate = new Date(article.processed_at || article.crawledAt);
    
    // Include articles published today OR processed today
    return (pubDate >= todayStart && pubDate <= todayEnd) || 
           (processedDate >= todayStart && processedDate <= todayEnd);
  });
  
  const dailyData = {
    ...rawData,
    articles: todaysArticles,
    processedAt: new Date().toISOString(),
    totalArticles: todaysArticles.length,
    categories: [...new Set(todaysArticles.map(a => a.category))],
    processingMethod: useAI ? 'ai-powered' : 'rule-based',
    newArticlesProcessed: newlyProcessedArticles.length,
    dailyArticlesCount: todaysArticles.length,
    filterDate: today
  };
  
  // Save files
  const latestPath = path.join(__dirname, '../data/latest-processed.json');
  const datePath = path.join(__dirname, `../data/${today}-processed.json`);
  
  // Save latest file (for current site build) - contains ALL articles
  await fs.writeFile(latestPath, JSON.stringify(latestData, null, 2));
  
  // Save date-specific file (for historical tracking) - contains ONLY today's articles
  await fs.writeFile(datePath, JSON.stringify(dailyData, null, 2));
  
  // Save updated rejected articles cache if we have new rejections
  if (newlyRejectedArticles.length > 0) {
    let allRejectedArticles = [...rejectedArticles.articles, ...newlyRejectedArticles];
    
    // Clean up old rejected articles (align with 15-day rolling archive) to prevent cache bloat
    const cleanupThresholdDays = parseInt(process.env.REJECTED_CACHE_CLEANUP_DAYS || '15');
    const cleanupThreshold = new Date(Date.now() - cleanupThresholdDays * 24 * 60 * 60 * 1000);
    const beforeCleanup = allRejectedArticles.length;
    
    allRejectedArticles = allRejectedArticles.filter(article => {
      const rejectedDate = new Date(article.rejected_at);
      return rejectedDate > cleanupThreshold;
    });
    
    const cleanedUp = beforeCleanup - allRejectedArticles.length;
    
    const rejectedOutputData = {
      articles: allRejectedArticles,
      updatedAt: new Date().toISOString(),
      totalRejected: allRejectedArticles.length,
      newlyRejected: newlyRejectedArticles.length,
      cleanupThresholdDays: cleanupThresholdDays,
      cleanedUpCount: cleanedUp
    };
    
    await fs.writeFile(rejectedDataPath, JSON.stringify(rejectedOutputData, null, 2));
    console.log(`🚫 Updated rejected articles cache: ${newlyRejectedArticles.length} new rejections, ${allRejectedArticles.length} total`);
    if (cleanedUp > 0) {
      console.log(`🧹 Cleaned up ${cleanedUp} rejected articles older than ${cleanupThresholdDays} days`);
    }
  }
  
  console.log(`✅ Successfully processed ${newlyProcessedArticles.length} new articles with ${useAI ? 'AI' : 'rule-based'} analysis`);
  console.log(`📊 Total articles in 15-day window: ${recentArticles.length} (${existingProcessed.articles.length} existing + ${newlyProcessedArticles.length} new)`);
  if (cleanedUpCount > 0) {
    console.log(`🧹 Cleaned up ${cleanedUpCount} articles older than ${cleanupThresholdDays} days`);
  }
  console.log(`🎯 New categories found:`, Object.keys(categoryStats).join(', ') || 'none');
  if (rejectedLowConfidence > 0) {
    const threshold = parseFloat(process.env.PROCESS_CONFIDENCE_THRESHOLD || '0.25');
    console.log(`🚫 Rejected ${rejectedLowConfidence} articles with confidence < ${(threshold * 100).toFixed(0)}%`);
  }
  console.log(`💾 Saved to: ${latestPath} (${recentArticles.length} articles in 15-day rolling window)`);
  console.log(`📅 Historical backup: ${datePath} (${todaysArticles.length} today's articles)`);
  console.log('🎉 Digest tagging completed successfully!');
  
  return latestData;
}

// Run if called directly
if (import.meta.url === `file://${__filename}`) {
  processArticlesWithAI().catch(console.error);
}

export default processArticlesWithAI; 