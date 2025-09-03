import { createClient } from '@/lib/supabase'
import type { Database } from '@/lib/supabase'

type Comedian = Database['public']['Tables']['comedians']['Row']
type UserProfile = Database['public']['Tables']['user_profiles']['Row']
type Interaction = Database['public']['Tables']['user_interactions']['Row']

interface RecommendationScore {
  comedian: Comedian
  score: number
  reasons: string[]
}

export class RecommendationEngine {
  private supabase = createClient()

  /**
   * Calculate cosine similarity between two vectors
   */
  private cosineSimilarity(vectorA: Record<string, number>, vectorB: Record<string, number>): number {
    const keysA = Object.keys(vectorA)
    const keysB = Object.keys(vectorB)
    const commonKeys = keysA.filter(key => keysB.includes(key))
    
    if (commonKeys.length === 0) return 0

    let dotProduct = 0
    let normA = 0
    let normB = 0

    for (const key of commonKeys) {
      const valueA = vectorA[key] || 0
      const valueB = vectorB[key] || 0
      dotProduct += valueA * valueB
      normA += valueA * valueA
      normB += valueB * valueB
    }

    if (normA === 0 || normB === 0) return 0
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
  }

  /**
   * Get user's interaction history to understand preferences
   */
  private async getUserInteractionProfile(userId: string): Promise<Record<string, number>> {
    const { data: interactions } = await this.supabase
      .from('user_interactions')
      .select(`
        comedian_id,
        interaction_type,
        rating,
        comedians!inner(humor_vector)
      `)
      .eq('user_id', userId)

    if (!interactions || interactions.length === 0) return {}

    const weightedProfile: Record<string, number> = {}
    const weights = {
      'like': 1.0,
      'save': 1.5,
      'dislike': -0.5,
      'skip': -0.2
    }

    for (const interaction of interactions) {
      const weight = weights[interaction.interaction_type as keyof typeof weights] || 0
      const ratingMultiplier = interaction.rating ? interaction.rating / 3 : 1 // Normalize rating
      const finalWeight = weight * ratingMultiplier

      // Get comedian's humor vector
      const comedianVector = (interaction as any).comedians.humor_vector as Record<string, number>
      
      for (const [dimension, value] of Object.entries(comedianVector)) {
        weightedProfile[dimension] = (weightedProfile[dimension] || 0) + (value * finalWeight)
      }
    }

    // Normalize the profile
    const maxValue = Math.max(...Object.values(weightedProfile).map(Math.abs))
    if (maxValue > 0) {
      for (const key of Object.keys(weightedProfile)) {
        weightedProfile[key] = weightedProfile[key] / maxValue
      }
    }

    return weightedProfile
  }

  /**
   * Apply demographic and contextual filters
   */
  private applyContextualBoosts(
    baseScore: number,
    comedian: Comedian,
    userProfile: UserProfile,
    context?: { timeOfDay?: string; weather?: string }
  ): { score: number; reasons: string[] } {
    let adjustedScore = baseScore
    const reasons: string[] = []

    // Location-based boost (if comedian performs locally)
    if (userProfile.location_state && comedian.popularity_score) {
      // This would require venue/tour data - simplified for now
      adjustedScore *= 1.1
      reasons.push('Popular in your area')
    }

    // Time-based adjustments
    if (context?.timeOfDay === 'evening') {
      // Evening might favor more mature content
      const edgyScore = (comedian.humor_vector as Record<string, number>)?.edgy || 0
      if (edgyScore > 0.6) {
        adjustedScore *= 1.2
        reasons.push('Great for evening entertainment')
      }
    }

    // Popularity boost for new users (low confidence)
    if ((userProfile.confidence_score || 0) < 0.3) {
      adjustedScore = adjustedScore * 0.7 + (comedian.popularity_score || 0.5) * 0.3
      reasons.push('Popular choice')
    }

    return { score: adjustedScore, reasons }
  }

  /**
   * Generate recommendations for a user
   */
  async getRecommendations(
    userId: string,
    limit: number = 10,
    context?: { timeOfDay?: string; weather?: string }
  ): Promise<RecommendationScore[]> {
    // Get user profile
    const { data: userProfile } = await this.supabase
      .from('user_profiles')
      .select('*')
      .eq('id', userId)
      .single()

    if (!userProfile) {
      throw new Error('User profile not found')
    }

    // Get all active comedians
    const { data: comedians } = await this.supabase
      .from('comedians')
      .select('*')
      .eq('is_active', true)

    if (!comedians) return []

    // Get user's interaction-based preferences
    const interactionProfile = await this.getUserInteractionProfile(userId)

    // Get comedians user has already interacted with (to avoid duplicates)
    const { data: existingInteractions } = await this.supabase
      .from('user_interactions')
      .select('comedian_id')
      .eq('user_id', userId)

    const interactedComedianIds = new Set(
      existingInteractions?.map(i => i.comedian_id) || []
    )

    // Calculate recommendations
    const recommendations: RecommendationScore[] = []

    for (const comedian of comedians) {
      // Skip comedians user has already interacted with
      if (interactedComedianIds.has(comedian.id)) continue

      let baseScore = 0
      const reasons: string[] = []

      // Use interaction-based profile if available, otherwise use survey profile
      const profileToUse = Object.keys(interactionProfile).length > 0 
        ? interactionProfile 
        : userProfile.humor_dimensions as Record<string, number>

      if (Object.keys(profileToUse).length > 0) {
        baseScore = this.cosineSimilarity(
          profileToUse,
          comedian.humor_vector as Record<string, number>
        )
        reasons.push('Matches your taste profile')
      }

      // Apply contextual boosts
      const { score: adjustedScore, reasons: additionalReasons } = this.applyContextualBoosts(
        baseScore,
        comedian,
        userProfile,
        context
      )

      recommendations.push({
        comedian,
        score: adjustedScore,
        reasons: [...reasons, ...additionalReasons]
      })
    }

    // Sort by score and return top recommendations
    return recommendations
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
  }

  /**
   * Record user interaction with a comedian
   */
  async recordInteraction(
    userId: string,
    comedianId: string,
    interactionType: 'like' | 'dislike' | 'save' | 'skip',
    rating?: number
  ): Promise<void> {
    await this.supabase.from('user_interactions').insert({
      user_id: userId,
      comedian_id: comedianId,
      interaction_type: interactionType,
      rating
    })

    // Update user's confidence score
    const { data: interactions } = await this.supabase
      .from('user_interactions')
      .select('id')
      .eq('user_id', userId)

    const interactionCount = interactions?.length || 0
    const newConfidence = Math.min(0.9, 0.1 + (interactionCount * 0.1))

    await this.supabase
      .from('user_profiles')
      .update({ 
        confidence_score: newConfidence,
        updated_at: new Date().toISOString()
      })
      .eq('id', userId)
  }

  /**
   * Get similar comedians to a given comedian
   */
  async getSimilarComedians(comedianId: string, limit: number = 5): Promise<Comedian[]> {
    const { data: targetComedian } = await this.supabase
      .from('comedians')
      .select('*')
      .eq('id', comedianId)
      .single()

    if (!targetComedian) return []

    const { data: allComedians } = await this.supabase
      .from('comedians')
      .select('*')
      .eq('is_active', true)
      .neq('id', comedianId)

    if (!allComedians) return []

    const similarities = allComedians.map(comedian => ({
      comedian,
      similarity: this.cosineSimilarity(
        targetComedian.humor_vector as Record<string, number>,
        comedian.humor_vector as Record<string, number>
      )
    }))

    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)
      .map(s => s.comedian)
  }
}