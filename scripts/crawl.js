import Parser from 'rss-parser';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { pipeline, env } from '@xenova/transformers';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configure transformers
env.allowRemoteFiles = true;
env.allowLocalFiles = true;
env.cacheDir = path.join(__dirname, '../.cache');

const parser = new Parser({
  timeout: 20000,
  maxRedirects: 5,
  headers: {
    'User-Agent': 'Mozilla/5.0 (compatible; MTMDigest/1.0; +https://github.com/deflogicals/mtm-daily-news.github.io)',
    'Accept': 'application/rss+xml, application/xml;q=0.9,*/*;q=0.8'
  }
});

let relevanceClassifier = null;

// Zero-shot classifier for digest relevance (Maharashtra / MSME / jobs / policy)
async function initializeRelevanceFilter() {
  try {
    console.log('🧠 Loading relevance classifier for digest filtering...');
    relevanceClassifier = await pipeline('zero-shot-classification', 'Xenova/distilbert-base-uncased-mnli', {
      cache_dir: env.cacheDir,
      quantized: true
    });
    console.log('✅ Relevance classifier ready');
    return true;
  } catch (error) {
    console.log('⚠️ Classifier failed to load, falling back to keyword filtering:', error.message);
    return false;
  }
}

// Zero-shot relevance: policy, business, jobs, Maharashtra vs fluff
async function isDigestRelevantAI(title, description = '') {
  if (!relevanceClassifier) return null;
  
  try {
    const text = `${title} ${description}`.substring(0, 500);
    
    const binaryCategories = [
      'News useful for India business policy government jobs education taxes or Maharashtra',
      'Pure entertainment sports celebrity gossip or lifestyle without economic or policy angle',
      'Unrelated memes games fiction or foreign celebrity gossip only'
    ];
    
    const binaryResult = await relevanceClassifier(text, binaryCategories);
    const isRelevant = binaryResult.scores[0] > Math.max(binaryResult.scores[1], binaryResult.scores[2]);
    const relConfidence = binaryResult.scores[0];
    
    let specificCategory = 'general policy and economy';
    let specificConfidence = relConfidence;
    
    if (isRelevant) {
      const specificCategories = [
        'Maharashtra Mumbai Pune state and local governance',
        'MSME small business industry credit and startups',
        'Government schemes subsidies and welfare programs',
        'Jobs recruitment exams and career announcements',
        'Education admissions scholarships and skills',
        'Tax GST banking RBI regulation and compliance',
        'National parliament courts laws and legal updates'
      ];
      
      const specificResult = await relevanceClassifier(text, specificCategories);
      specificCategory = specificResult.labels[0];
      specificConfidence = Math.max(specificConfidence, specificResult.scores[0]);
    }
    
    const confidenceThreshold = parseFloat(process.env.CATEGORIZATION_CONFIDENCE_THRESHOLD || '0.22');
    const meetsQualityThreshold = relConfidence >= confidenceThreshold;
    
    if (isRelevant && !meetsQualityThreshold) {
      console.log(`🚫 Quality filtered: "${title.substring(0, 50)}..." (confidence: ${(relConfidence * 100).toFixed(1)}% < ${(confidenceThreshold * 100).toFixed(0)}%)`);
    } else if (!isRelevant && relConfidence > 0.2) {
      console.log(`❌ Filtered out: "${title}" (confidence: ${relConfidence.toFixed(2)}, bucket: ${binaryResult.labels[0]})`);
    }
    
    return {
      isRelevant,
      confidence: relConfidence,
      topCategory: specificCategory,
      meetsQualityThreshold
    };
  } catch (error) {
    console.log('Relevance classification error:', error.message);
    return null;
  }
}

// Load sources
async function loadSources() {
  const sourcesPath = path.join(__dirname, '../sources.json');
  const sourcesData = await fs.readFile(sourcesPath, 'utf-8');
  const { 
    sources, 
    medium_blogs,
    article_sources, 
    youtube_channels, 
    newsletters, 
    developer_blogs,
    academic_sources,
    job_opportunity
  } = JSON.parse(sourcesData);
  
  return [
    ...sources,
    ...(medium_blogs || []),
    ...(article_sources || []),
    ...(youtube_channels || []),
    ...(newsletters || []),
    ...(developer_blogs || []),
    ...(academic_sources || []),
    ...(job_opportunity || [])
  ];
}

// Extract domain from URL
function extractDomain(url) {
  try {
    return new URL(url).hostname.replace('www.', '');
  } catch {
    return 'unknown';
  }
}

// Clean and normalize title
function cleanTitle(title) {
  return title
    .replace(/\[.*?\]/g, '') // Remove [tags]
    .replace(/\(.*?\)/g, '') // Remove (parentheses) 
    .replace(/\s+/g, ' ')    // Normalize whitespace
    .trim();
}

function isDigestRelatedItem(item) {
  const text = `${item.title || ''} ${item.description || ''} ${item.content || ''}`.toLowerCase();
  
  const excludeKeywords = [
    'ipl ', ' ipl', 'world cup', 't20', 'bollywood', 'movie review', 'celebrity wedding',
    'web series', 'bigg boss', 'recipe ', 'horoscope'
  ];
  
  if (excludeKeywords.some(keyword => text.includes(keyword))) {
    return false;
  }
  
  const keywords = [
    'maharashtra', 'mumbai', 'pune', 'nagpur', 'nashik', 'aurangabad', 'thane',
    'msme', 'udyam', 'sme', 'startup', 'gst', 'income tax', 'tax ', ' rbi', 'sebi',
    'scheme', 'subsidy', 'tender', 'recruitment', 'vacancy', 'job ', 'jobs ', 'exam ',
    'admission', 'scholarship', 'university', 'skill ', 'policy', 'cabinet', 'parliament',
    'bill ', 'ordinance', 'high court', 'supreme court', 'minister', 'government',
    'notification', 'circular', 'loan ', 'credit ', 'bank ', 'economy', 'budget',
    'export', 'import', 'industry', 'factory', 'midc', 'midc', 'fiscal', 'fdi',
    'regulation', 'compliance', 'employees', 'labour', 'labor', 'wage', 'pension',
    'farmers', 'agriculture', 'power sector', 'infrastructure'
  ];
  
  return keywords.some(keyword => text.includes(keyword));
}

function isDigestRelevantTitle(title, source) {
  const src = source.toLowerCase();
  const trusted = [
    'pib', 'rbi', 'prs', 'dd news', 'lokmat', 'esakal', 'pudhari', 'tarun bharat',
    'nagpur today', 'live nagpur', 'punekar', 'mumbai live', 'navakal', 'deshdoot',
    'dainik ekmat', 'nagar live', 'jalgaon', 'beed reporter', 'news18',
    'times now marathi', 'tak.live', 'agrowon',
    'freshersvoice', 'pagalguy', 'yourstory', 'inc42', 'reddit', 'pib india', 'aaple sarkar'
  ];
  if (trusted.some(t => src.includes(t))) {
    return true;
  }
  const titleLower = title.toLowerCase();
  const keywords = [
    'maharashtra', 'mumbai ', 'pune', 'nagpur',
    'msme', 'udyam', 'sme ', 'startup', 'gst', 'income tax', 'tax ', 'rbi', 'sebi',
    'scheme', 'subsidy', 'tender', 'recruitment', 'vacancy', 'job ', 'jobs ', 'exam ',
    'admission', 'scholarship', 'policy', 'cabinet', 'parliament',
    'bill ', 'court', 'minister', 'government', 'notification', 'loan ', ' bank',
    'economy', 'budget', 'export', 'industry', 'regulation', 'compliance',
    'employees', 'labour', 'labor', 'wage', 'pension', 'india ', 'indian ',
    'education', 'university', 'college', 'skill india', 'skill development'
  ];
  return keywords.some(k => titleLower.includes(k));
}

// Main crawl function
async function crawlAllSources() {
  console.log('📰 Starting MTM digest crawl...');
  
  // Initialize zero-shot relevance filter
  const relevanceFilterReady = await initializeRelevanceFilter();
  
  const sources = await loadSources();
  const priorityRank = { highest: 0, high: 1, medium: 2, low: 3 };
  sources.sort((a, b) =>
    (priorityRank[a.priority] ?? 2) - (priorityRank[b.priority] ?? 2)
  );
  console.log(`Found ${sources.length} sources to crawl (sorted: highest → low priority)`);
  
  // Crawl all sources in parallel (but with some delay to be nice)
  const allArticles = [];
  const crawlStats = { totalProcessed: 0, qualityFiltered: 0, irrelevantFiltered: 0 };
  const batchSize = 5; // Process 5 sources at a time
  
  for (let i = 0; i < sources.length; i += batchSize) {
    const batch = sources.slice(i, i + batchSize);
    const promises = batch.map(source => crawlFeed(source, relevanceFilterReady, crawlStats));
    const results = await Promise.all(promises);
    
    for (const result of results) {
      if (result.articles) {
        allArticles.push(...result.articles);
      }
    }
    
    // Small delay between batches
    if (i + batchSize < sources.length) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  // Remove duplicates
  const uniqueArticles = removeDuplicates(allArticles);
  
  // Sort by publication date (newest first)
  uniqueArticles.sort((a, b) => new Date(b.pubDate) - new Date(a.pubDate));
  
  console.log(`📰 Found ${uniqueArticles.length} unique digest articles`);
  console.log(`🧠 Relevance filter: ${relevanceFilterReady ? 'ENABLED' : 'Keyword fallback'}`);
  if (relevanceFilterReady && crawlStats.qualityFiltered > 0) {
    const threshold = parseFloat(process.env.CRAWL_CONFIDENCE_THRESHOLD || '0.30');
    console.log(`🚫 Quality filtered during crawl: ${crawlStats.qualityFiltered} articles (< ${(threshold * 100).toFixed(0)}% confidence)`);
  }
  if (relevanceFilterReady && crawlStats.irrelevantFiltered > 0) {
    console.log(`❌ Relevance filtered out: ${crawlStats.irrelevantFiltered} articles`);
  }
  console.log(`📊 Crawl stats: ${crawlStats.totalProcessed} processed → ${uniqueArticles.length} kept`);
  
  // Ensure data directory exists
  const dataDir = path.join(__dirname, '../data');
  await fs.mkdir(dataDir, { recursive: true });
  
  // Save raw crawled data
  const output = {
    crawledAt: new Date().toISOString(),
    totalSources: sources.length,
    totalArticles: uniqueArticles.length,
    relevanceFilterUsed: relevanceFilterReady,
    articles: uniqueArticles
  };
  
  // Save only as latest-raw.json (no dated duplicates)
  const filepath = path.join(dataDir, 'latest-raw.json');
  await fs.writeFile(filepath, JSON.stringify(output, null, 2));
  console.log(`💾 Saved raw data to: latest-raw.json`);
  
  return uniqueArticles;
}

// Crawl a single RSS feed with relevance filtering
async function crawlFeed(source, useRelevanceFilter = false, stats = null) {
  try {
    console.log(`Crawling: ${source.name}`);
    
    const feed = await parser.parseURL(source.url);
    const articles = [];
    
    // Different limits based on source type
    const itemLimit = source.category === 'marathi' ? 24 :
                     source.category === 'local_mh' ? 20 :
                     source.category === 'reddit' ? 15 :
                     source.category === 'youtube' ? 10 :
                     source.category === 'newsletter' ? 12 :
                     source.category === 'startup' ? 12 :
                     source.category === 'community' ? 12 :
                     source.category === 'jobs' ? 22 :
                     source.category === 'education' ? 18 :
                     source.category === 'state_mh' ? 20 :
                     source.category === 'national' ? 16 :
                     source.category === 'legal' ? 15 :
                     source.category === 'msme' ? 18 :
                     source.category === 'business' ? 16 :
                     source.category === 'news' ? 14 : 14;
    
    const items = feed.items.slice(0, itemLimit);
    
    for (const item of items) {
      const title = cleanTitle(item.title || '');
      const url = item.link || item.guid;
      
      if (!title || !url) continue;
      
      if (stats) stats.totalProcessed++;
      
      // Extract description for relevance check
      let description = '';
      if (item.contentSnippet) {
        description = item.contentSnippet.substring(0, 200);
      } else if (item.content) {
        description = item.content.replace(/<[^>]*>/g, '').substring(0, 200);
      } else if (item.summary) {
        description = item.summary.replace(/<[^>]*>/g, '').substring(0, 200);
      }
      
      // Classifier (preferred) or keyword fallback
      let isRelevant = false;
      
      if (useRelevanceFilter) {
        // Use AI classifier for intelligent filtering with 75% confidence threshold
        const aiResult = await isDigestRelevantAI(title, description);
        
        if (aiResult) {
        
          // Apply confidence threshold for quality filtering
          isRelevant = aiResult.isRelevant && aiResult.meetsQualityThreshold;
          
          // Log AI decisions for debugging and track stats
          if (aiResult.isRelevant && !aiResult.meetsQualityThreshold) {
            const threshold = parseFloat(process.env.CRAWL_CONFIDENCE_THRESHOLD || '0.30');
            console.log(`🚫 Quality filtered: "${title.substring(0, 50)}..." (confidence: ${(aiResult.confidence * 100).toFixed(1)}% < ${(threshold * 100).toFixed(0)}%)`);
            if (stats) stats.qualityFiltered++;
          } else if (!aiResult.isRelevant && aiResult.confidence > 0.2) {
            console.log(`❌ Filtered out: "${title}" (confidence: ${aiResult.confidence.toFixed(2)}, category: ${aiResult.topCategory})`);
            if (stats) stats.irrelevantFiltered++;
          }
        } else {
          // AI failed, fallback to keyword filtering
          isRelevant = isDigestRelevantTitle(title, source.name);
        }
      } else {
        // Fallback keyword filtering with improved logic
        const basicDigest = isDigestRelevantTitle(title, source.name);
        
        if (source.category === 'jobs' || source.category === 'education') {
          const broad = [
            'job', 'exam', 'result', 'admission', 'recruit', 'vacancy', 'course',
            'scholarship', 'university', 'college', 'skill', 'career', 'trainee'
          ];
          const broadHit = broad.some(k => title.toLowerCase().includes(k.trim()));
          isRelevant = basicDigest || broadHit;
        } else if (source.category === 'reddit') {
          const rkeys = [
            'job', 'hiring', 'recruit', 'vacancy', 'exam', 'government', 'policy', 'scheme',
            'mumbai', 'pune', 'maharashtra', 'thane', 'nagpur', 'nashik', 'msme', 'tax', 'metro', 'bmc'
          ];
          isRelevant = basicDigest || rkeys.some(k => title.toLowerCase().includes(k));
        } else {
          isRelevant = basicDigest;
        }
        
        const excludeTerms = [
          'ipl ', ' ipl', 't20 world cup', 'match preview', 'bollywood', 'box office',
          'movie review', 'celebrity', 'bigg boss', 'web series trailer'
        ];
        const hasExcludedTerm = excludeTerms.some(term => 
          title.toLowerCase().includes(term)
        );
        
        if (hasExcludedTerm) {
          isRelevant = false;
        }
      }
      
      if (!isRelevant) continue;
      
      // Different time windows based on source type (max 15 days to align with cleanup)
      const daysBack = source.category === 'marathi' ? 5 :
                      source.category === 'local_mh' ? 7 :
                      source.category === 'reddit' ? 4 :
                      source.category === 'youtube' ? 14 :
                      source.category === 'newsletter' ? 7 :
                      source.category === 'startup' ? 7 :
                      source.category === 'community' ? 5 :
                      source.category === 'jobs' ? 10 :
                      source.category === 'education' ? 10 :
                      source.category === 'state_mh' ? 7 :
                      source.category === 'national' ? 7 :
                      source.category === 'legal' ? 14 :
                      source.category === 'msme' ? 7 :
                      source.category === 'business' ? 7 :
                      source.category === 'news' ? 5 : 7;
      
      const pubDate = new Date(item.pubDate || item.isoDate || item.published || Date.now());
      
      // Validate date - skip articles with invalid or future dates
      if (isNaN(pubDate.getTime())) {
        console.log(`⚠️ Invalid date for article: "${title.substring(0, 50)}..."`);
        continue;
      }
      
      const now = new Date();
      if (pubDate > now) {
        console.log(`⚠️ Future date detected for article: "${title.substring(0, 50)}..." (${pubDate.toISOString()})`);
        // Use current time instead of future date
        pubDate.setTime(now.getTime());
      }
      
      const cutoffDate = new Date(Date.now() - daysBack * 24 * 60 * 60 * 1000);
      if (pubDate < cutoffDate) continue;
      
      articles.push({
        title: title,
        url: url,
        source: source.name,
        source_domain: extractDomain(url),
        source_category: source.category,
        source_priority: source.priority,
        pubDate: pubDate.toISOString(),
        metaDescription: description,
        
        // Will be filled by LLM processing
        category: null,
        difficulty: null,
        confidence: null,
        
        // Metadata
        crawledAt: new Date().toISOString(),
        id: generateId(title, url)
      });
    }
    
    console.log(`✓ ${source.name}: ${articles.length} articles kept`);
    return { articles, stats };
    
  } catch (error) {
    console.error(`✗ Failed to crawl ${source.name}:`, error.message);
    return { articles: [], stats };
  }
}

// Generate unique ID for article
function generateId(title, url) {
  const content = title + url;
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(36);
}

// Remove duplicates based on similarity
function removeDuplicates(articles) {
  const unique = [];
  const seen = new Set();
  
  for (const article of articles) {
    // Create a normalized key for duplicate detection
    const key = article.title
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
      .split(' ')
      .slice(0, 8) // First 8 words
      .join(' ');
    
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(article);
    }
  }
  
  return unique;
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  crawlAllSources()
    .then(articles => {
      console.log(`✅ Crawl complete! Found ${articles.length} articles`);
    })
    .catch(error => {
      console.error('❌ Crawl failed:', error);
      process.exit(1);
    });
}

export { crawlAllSources }; 