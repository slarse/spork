package se.kth.spork.spoon;

import se.kth.spork.base3dm.Pcs;
import se.kth.spork.base3dm.Revision;
import se.kth.spork.spoon.wrappers.NodeFactory;
import se.kth.spork.spoon.wrappers.SpoonNode;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.visitor.CtScanner;

import java.util.*;

/**
 * A scanner that builds a PCS structure from a Spoon tree.
 *
 * @author Simon Larsén
 */
class PcsBuilder extends CtScanner {
    private Map<SpoonNode, SpoonNode> rootTolastSibling = new HashMap<>();
    private Set<Pcs<SpoonNode>> pcses = new HashSet<>();
    private SpoonNode root = null;
    private Revision revision;

    public PcsBuilder(Revision revision) {
       super();
       this.revision = revision;
    }

    /**
     * Convert a Spoon tree into a PCS structure.
     *
     * @param spoonClass A Spoon class.
     * @param revision   The revision this Spoon class belongs to. The revision is attached to each PCS triple.
     * @return The Spoon tree represented by PCS triples.
     */
    public static Set<Pcs<SpoonNode>> fromSpoon(CtElement spoonClass, Revision revision) {
        PcsBuilder scanner = new PcsBuilder(revision);
        scanner.scan(spoonClass);
        return scanner.getPcses();
    }

    @Override
    protected void enter(CtElement e) {
        SpoonNode wrapped = NodeFactory.wrap(e);
        if (root == null)
            root = wrapped;

        SpoonNode parent = wrapped.getParent();
        SpoonNode predecessor = rootTolastSibling.getOrDefault(parent, NodeFactory.startOfChildList(wrapped.getParent()));
        pcses.add(new Pcs(parent, predecessor, wrapped, revision));
        rootTolastSibling.put(parent, wrapped);
    }

    @Override
    protected void exit(CtElement e) {
        if (e == root.getElement()) {
            finalizePcsLists();
        }
    }

    /**
     * Add the last element to each PCS list (i.e. Pcs(root, child, null)).
     */
    private void finalizePcsLists() {
        for (SpoonNode predecessor : rootTolastSibling.values()) {
            pcses.add(new Pcs(predecessor.getParent(), predecessor, NodeFactory.endOfChildList(predecessor.getParent()), revision));
        }
    }

    public Set<Pcs<SpoonNode>> getPcses() {
        return pcses;
    }
}
